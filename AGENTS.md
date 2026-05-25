# Guara Vivo Identifier - Instruções para Agentes

## Contexto do Projeto

**Objetivo:** API FastAPI que detecta e classifica guaras-vermelhas em imagens usando YOLO, MobileNetV2 e análise de cor com KMeans.

**Stack:** Python 3.9+, FastAPI, TensorFlow, OpenCV, NumPy

**Arquivos principais:**
- `src/api.py` — endpoint de inferência com validações de segurança
- `src/services/guara_identifier_service.py` — orquestração do pipeline
- `src/classifiers/species_classifier.py` — classificação de espécie com DRY refactoring
- `src/analyzers/` — análise de cor e estimativa de distância
- `README.md` — documentação completa do pipeline

---

## Últimas Alterações (Commits)

### 1. `fix(api): Add upload validation, size limits, and thread pool for inference`
**Impacto:** Segurança + Performance crítica

- ✅ **Validação MIME:** Apenas `image/jpeg`, `image/jpg`, `image/png` aceitos (HTTP 415)
- ✅ **Limite de tamanho:** Máximo 10 MB por upload (HTTP 413 se excedido)
- ✅ **Thread pool:** Inferência pesada (YOLO + MobileNet) roda em `run_in_threadpool` para não bloquear event loop
- ✅ **Segurança de erro:** Exceções internas não são retornadas ao cliente (resposta genérica + log interno)
- ✅ **Logging estruturado:** Erros capturam stack trace, avisos registram apenas mensagem

**Constantes em `src/api.py:13-17`:**
```python
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THREAD_POOL_WORKERS = 2  # Workers para inferência pesada
```

### 2. `refactor(classifier): Extract duplicate prediction logic and improve .gitignore`
**Impacto:** Manutenibilidade + Governança de repositório

- ✅ **Helper `_predict_probs()`:** Extrai preprocessing duplicado (resize → expand_dims → preprocess_input → predict)
  - Reutilizado por `classify()` e `guara_confidence()`
  - Reduz risco de divergência futura
  
- ✅ **`.gitignore` robusto:**
  - Virtual environments: `venv/`, `.venv/`, `env/`, `.env`
  - Python: `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.mypy_cache/`
  - Models: `*.pt`, `*.pth`, `*.h5`, `*.pb`, `*.onnx`
  - IDE: `.vscode/`, `.idea/`
  - OS: `.DS_Store`, `Thumbs.db`

---

## Orientações para Agentes

### Segurança
- **Nunca commit secrets:** `.env` está ignorado; use variáveis de ambiente
- **Upload:** Validar MIME + extensão + tamanho SEMPRE antes de processar
- **Erro:** Log com stack trace internamente, resposta genérica ao cliente
- **Dependencies:** Pin versões em `requirements.txt`

### Performance
- **Inferência pesada:** Deve rodas em thread pool (`run_in_threadpool`)
- **Bloqueio:** Event loop nunca deve estar bloqueado por CV/ML
- **Concorrência:** `THREAD_POOL_WORKERS=2` é padrão; ajuste conforme carga esperada

### Code Quality
- **DRY:** Extrair helpers antes de duplicação
- **Tipos:** Usar type hints em assinaturas (ex: `def _predict_probs(self, crop_rgb: np.ndarray) -> np.ndarray`)
- **Logging:** Usar `logger` do módulo, nunca `print`
