# Guara Vivo Identifier

Este projeto detecta passaros em uma imagem com YOLO, identifica quais deteccoes sao de guara com MobileNetV2, e classifica a cor de cada guara para inferir fase de vida:

- vermelho -> adulto
- cinza -> filhote

Tambem retorna as acuracias de cada etapa para cada guara encontrado.

O pipeline usa modo hibrido:

- Bounding box para classificacao de especie (MobileNetV2)
- Mascara de silhueta para analise de cor, quando disponivel
- Fallback automatico para bounding box quando a mascara nao for confiavel

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
   - resultado por guara (bbox, cor, fase de vida)
  - origem da analise de cor (mask ou bbox fallback)
  - acuracia por etapa (YOLO, classificacao de guara, cor/fase)

## Requisitos

Os pacotes estao em requirements.txt.

## Como executar

No terminal, dentro da pasta do projeto:

```bash
python main.py --image guara4.jpg
```

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
      "acuracia": {
        "deteccao_yolo": 0.9221,
        "classificacao_guara": 0.9572,
        "classificacao_cor": 0.8112,
        "classificacao_fase_vida": 0.8112
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
2. src/analyzers/color_analyzer.py:
   - regra de classificacao no metodo _classify_color
  - regras de fallback de mascara no metodo _extract_color_pixels
  - referencias de vermelho e cinza

Assim, o restante do pipeline (YOLO + MobileNet + saida final) nao precisa ser alterado.

## Observacoes

- A acuracia da fase de vida usa a mesma confianca da classificacao de cor, porque fase de vida e derivada diretamente da cor.
- Se o recorte for invalido, a classificacao pode retornar indefinido com confianca 0.0.
- Para usar silhuetas, o modelo YOLO precisa fornecer masks. Se nao houver masks no resultado, o sistema continua funcionando com bbox (fallback).
