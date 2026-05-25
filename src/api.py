import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.services.guara_identifier_service import GuaraIdentifierService

logger = logging.getLogger(__name__)

# Configurações de segurança
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THREAD_POOL_WORKERS = 2  # Workers para inferência pesada


def create_app() -> FastAPI:
    app = FastAPI(
        title="Guara Vivo Identifier API",
        description="API para inferencia de guara-vermelho em imagens enviadas na requisicao.",
        version="1.0.0",
    )

    service = GuaraIdentifierService()
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    def _validate_upload(image: UploadFile) -> tuple[str, bytes]:
        """Valida tipo MIME, extensão e tamanho do arquivo."""
        # Validar MIME type
        if image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Tipo de arquivo não suportado. Use: {', '.join(ALLOWED_MIME_TYPES)}",
            )
        
        # Validar extensão
        filename = image.filename or "unknown"
        file_ext = f".{filename.split('.')[-1].lower()}" if "." in filename else None
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Extensão de arquivo não permitida. Use: {', '.join(ALLOWED_EXTENSIONS)}",
            )
        
        return filename, None

    async def _process_inference(image_bgr: np.ndarray, filename: str) -> dict:
        """Executa inferência pesada em thread pool."""
        try:
            return await run_in_threadpool(
                service.process_image_bgr, image_bgr, filename
            )
        except ValueError as exc:
            logger.warning(f"Validação falhou: {exc}", exc_info=False)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(f"Erro interno durante inferência", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Erro interno durante a inferência. Por favor, tente novamente mais tarde.",
            ) from exc

    @app.post("/guara-vermelho/inference")
    async def inference(image: UploadFile = File(...)) -> dict:
        # Validar tipo e extensão
        _validate_upload(image)
        
        # Ler arquivo
        contents = await image.read()
        
        # Validar tamanho
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="A imagem enviada está vazia")
        if len(contents) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande. Máximo permitido: {MAX_IMAGE_SIZE_BYTES / 1024 / 1024:.0f} MB",
            )

        # Decodificar imagem
        image_array = np.frombuffer(contents, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(
                status_code=400,
                detail="Não foi possível decodificar a imagem enviada"
            )

        # Executar inferência em thread pool
        return await _process_inference(image_bgr, image.filename or "imagem_enviada")

    return app


app = create_app()