# Guara Vivo Identifier

Este projeto detecta passaros em uma imagem com YOLO, identifica quais deteccoes sao de guara com MobileNetV2, e classifica a cor de cada guara para inferir fase de vida:

- vermelho -> adulto
- cinza -> filhote

Tambem retorna as acuracias de cada etapa para cada guara encontrado.
Tambem estima a distancia de cada guara com base na altura/envergadura medias da especie.

O pipeline usa modo hibrido:

- Bounding box para classificacao de especie (MobileNetV2)
- Mascara de silhueta para analise de cor, quando disponivel
- Fallback automatico para bounding box quando a mascara nao for confiavel
- Estimativa de distancia por tamanho aparente em pixels

## Estrutura do projeto

- main.py: ponto de entrada da aplicacao
- src/config.py: configuracoes dos modelos e do analisador de cor
- src/types.py: classes de dados do pipeline
- src/detectors/yolo_detector.py: classe para deteccao de passaros
- src/classifiers/species_classifier.py: classe para classificacao de especie
- src/analyzers/color_analyzer.py: classe para KMeans e classificacao de cor
- src/services/guara_identifier_service.py: orquestracao completa do fluxo

## Como funciona

1. A classe YoloBirdDetector detecta objetos do tipo bird na imagem.
2. Cada deteccao vira um recorte individual.
3. A classe SpeciesClassifier classifica o recorte e filtra apenas guara_vermelho.
4. A classe GuaraColorAnalyzer roda KMeans nos pixels da mascara (silhueta) quando possivel.
5. Se a mascara nao existir ou tiver cobertura insuficiente, o sistema usa fallback para os pixels da bounding box.
6. A cor e classificada entre vermelho (adulto) ou cinza (filhote).
7. A classe GuaraIdentifierService monta a resposta final com:

- quantidade total de guaras
- resultado por guara (bbox, cor, fase de vida, distancia)
- origem da analise de cor (mask ou bbox fallback)
- acuracia por etapa (YOLO, classificacao de guara, cor/fase, distancia)

## Estimativa de distancia

A estimativa usa o modelo pinhole:

- distancia = (focal_px \* tamanho_real_cm) / tamanho_pixels

No projeto, os valores padrao sao:

- altura media adulta: 63.5 cm
- envergadura media adulta: 52-56 cm

O sistema calcula duas estimativas:

- por altura observada em pixels
- por largura observada em pixels (envergadura)

Depois escolhe automaticamente o metodo mais adequado pelo aspecto do objeto:

- wingspan: quando a largura domina
- height: quando a altura domina

A incerteza cresce quando os dois metodos discordam.

## Requisitos

Os pacotes estao em requirements.txt.

## Como executar

Suba a API no terminal, dentro da pasta do projeto:

```bash
uvicorn src.api:app --reload
```

Se preferir, tambem funciona:

```bash
python main.py
```

## Rota de inferencia

- `POST /guara-vermelho/inference`
- Envio da imagem: `multipart/form-data` com o campo `image`

Exemplo com `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/guara-vermelho/inference" \
  -F "image=@guara4.jpg"
```

A resposta retorna o mesmo JSON que antes era impresso no terminal.

## Exemplo de saida

```json
{
	"imagem": "guara4.jpg",
	"quantidade_guaras": 2,
	"guaras": [
		{
			"id": 1,
			"bbox_xyxy": [100, 60, 260, 310],
			"cor": "vermelho",
			"fase_vida": "adulto",
			"fonte_analise_cor": "mask",
			"cor_dominante_rgb": [214, 73, 58],
			"distancia_estimada_m": 8.42,
			"incerteza_distancia_m": 1.89,
			"metodo_distancia": "height",
			"fonte_pixels_distancia": "mask",
			"medidas_pixels_objeto": {
				"largura_px": 104,
				"altura_px": 146
			},
			"acuracia": {
				"deteccao_yolo": 0.9221,
				"classificacao_guara": 0.9572,
				"classificacao_cor": 0.8112,
				"classificacao_fase_vida": 0.8112,
				"estimativa_distancia": 0.79
			}
		}
	]
}
```

## Como ajustar apenas o KMeans

Se voce quiser mexer somente na parte de cor, a manutencao esta isolada em dois pontos:

1. src/config.py:
   - ColorAnalyzerConfig.kmeans_clusters
   - ColorAnalyzerConfig.resized_width
   - ColorAnalyzerConfig.resized_height

- ColorAnalyzerConfig.use_mask_for_color
- ColorAnalyzerConfig.min_mask_coverage
- ColorAnalyzerConfig.mask_erosion_kernel_size
- ColorAnalyzerConfig.min_mask_pixels
- DistanceEstimatorConfig.focal_length_px
- DistanceEstimatorConfig.adult_height_cm
- DistanceEstimatorConfig.adult_wingspan_cm_min
- DistanceEstimatorConfig.adult_wingspan_cm_max
- DistanceEstimatorConfig.wingspan_aspect_ratio_threshold
- DistanceEstimatorConfig.min_object_pixels
- DistanceEstimatorConfig.min_relative_uncertainty

2. src/analyzers/color_analyzer.py:
   - regra de classificacao no metodo \_classify_color

- regras de fallback de mascara no metodo \_extract_color_pixels
- referencias de vermelho e cinza

3. src/analyzers/distance_estimator.py:


    - regra de escolha de metodo (height/wingspan)
    - regra de confianca e incerteza da distancia

Assim, o restante do pipeline (YOLO + MobileNet + saida final) nao precisa ser alterado.

## Observacoes

- A acuracia da fase de vida usa a mesma confianca da classificacao de cor, porque fase de vida e derivada diretamente da cor.
- Se o recorte for invalido, a classificacao pode retornar indefinido com confianca 0.0.
- Para usar silhuetas, o modelo YOLO precisa fornecer masks. Se nao houver masks no resultado, o sistema continua funcionando com bbox (fallback).
- Distancia com imagem unica e aproximada; para maior precisao, calibre a camera (focal em pixels) no ambiente real de captura.
