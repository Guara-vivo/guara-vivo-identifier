import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.services.guara_identifier_service import GuaraIdentifierService


def create_app() -> FastAPI:
    app = FastAPI(
        title="Guara Vivo Identifier API",
        description="API para inferencia de guara-vermelho em imagens enviadas na requisicao.",
        version="1.0.0",
    )

    service = GuaraIdentifierService()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/guara-vermelho/inference")
    async def inference(image: UploadFile = File(...)) -> dict:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="A imagem enviada esta vazia")

        image_array = np.frombuffer(contents, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Nao foi possivel decodificar a imagem enviada")

        try:
            return service.process_image_bgr(image_bgr, image.filename or "imagem_enviada")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Erro interno durante a inferencia: {exc}") from exc

    return app


app = create_app()