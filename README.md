# Guará Vivo Identifier

Serviço de IA do Guará Vivo. Recebe uma foto e devolve quantos guarás-vermelhos aparecem nela, a fase de vida de cada um (adulto ou filhote) e uma estimativa de distância.

Funciona em etapas:

1. **YOLO** (`yolo11x-seg`) encontra as aves na imagem.
2. **MobileNetV2** (`modelo_guara_mobilenetv2.keras`) diz se cada ave é guará-vermelho, colhereiro ou garça.
3. **Análise de cor** (HSV na silhueta da ave) define a fase: vermelho → adulto, cinza → filhote. Quando não há sinal claro, a fase é `indefinido` e a resposta traz um aviso.
4. Estima a distância de cada guará pelo tamanho aparente em pixels.

Quem chama este serviço é o [worker](https://github.com/Guara-vivo/guara-vivo-worker), uma foto por requisição.

**Stack:** Python 3.12, FastAPI, TensorFlow/Keras, PyTorch + Ultralytics (YOLO), OpenCV, scikit-learn.

## O que você precisa antes

- **Python 3.12** (a versão usada no Docker). TensorFlow e PyTorch têm versão fixa no `requirements.txt`.
- Espaço livre para as dependências (TensorFlow e PyTorch são pesados, vários GB) e cerca de **1 GB de RAM** com os modelos carregados. Roda em CPU.
- O arquivo **`yolo11x-seg.pt`** na raiz do projeto. Ele não vem no Git (`*.pt` está no `.gitignore`).
  - Rodando em Python: se o arquivo não existir, o Ultralytics normalmente o baixa sozinho na primeira execução (~120 MB).
  - Para o build com Docker o arquivo **precisa existir antes**, porque o `Dockerfile` o copia. Baixe em https://github.com/ultralytics/assets/releases (`yolo11x-seg.pt`).
- O modelo `modelo_guara_mobilenetv2.keras` já vem no repositório.

## Rodando localmente

```powershell
python -m venv venv
.\venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
pip install --no-deps ultralytics==8.4.21
```

> O `ultralytics` fica fora do `requirements.txt` de propósito (o `Dockerfile` também o instala à parte, com `--no-deps`, para não puxar dependências duplicadas). Sem esse segundo `pip install` o serviço não sobe.

Suba a API:

```powershell
uvicorn src.api:app --port 8000 --reload
# ou: python main.py
```

A primeira subida é lenta porque carrega os modelos. Para conferir que está pronto: http://localhost:8000/health deve responder `{"status":"ok"}` (e há documentação interativa em http://localhost:8000/docs).

### Testando com uma imagem

```bash
curl -X POST "http://localhost:8000/guara-vermelho/inference" \
  -F "image=@caminho/para/foto.jpg"
```

- Aceita **JPEG ou PNG** de até **10 MB**, um arquivo por vez, no campo `image`.
- A resposta traz `status_analise`, `quantidade_guaras` e, para cada guará, `cor`, `fase_vida`, `acuracia` de cada etapa, a caixa na imagem (`bbox_xyxy`) e a distância estimada (`distancia_estimada_m`). Um exemplo completo da saída está em [`docs/REFERENCIA.md`](docs/REFERENCIA.md).

### Com Docker

```bash
docker build -t guara-vivo-identifier .
docker run -p 8000:8000 guara-vivo-identifier
```

(precisa do `yolo11x-seg.pt` na pasta antes do build, como explicado acima.)

## Testes

```powershell
pip install pytest
python -m pytest -q
```

Hoje cobrem a análise de cor e fase de vida.

## Estrutura

```
main.py                              sobe a API (uvicorn)
src/api.py                           endpoint /guara-vermelho/inference e /health
src/config.py                        caminhos dos modelos e limiares (cor, distância)
src/detectors/yolo_detector.py       detecção de aves (YOLO)
src/classifiers/species_classifier.py  classificação de espécie (MobileNetV2)
src/analyzers/                       análise de cor / fase de vida e distância
src/services/guara_identifier_service.py  orquestra o pipeline completo
tests/                               testes (pytest)
docs/                                documentação de referência
```

## Mais detalhes

Como a estimativa de distância funciona, formato completo da resposta, status e avisos da análise e como ajustar o KMeans estão em [`docs/REFERENCIA.md`](docs/REFERENCIA.md).
