"""A inferência roda no executor dedicado: no máximo THREAD_POOL_WORKERS de cada vez.

O serviço real (YOLO + TensorFlow) é trocado por um falso, então o teste não carrega modelos.
"""
import asyncio
import importlib
import io
import sys
import threading
import time
import types

import cv2
import numpy as np
from starlette.datastructures import Headers, UploadFile


class _FakeService:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.calls = 0

    def process_image_bgr(self, image_bgr, image_source="imagem_enviada"):
        with self.lock:
            self.running += 1
            self.calls += 1
            self.max_running = max(self.max_running, self.running)
        time.sleep(0.2)
        with self.lock:
            self.running -= 1
        return {"results": []}


def _load_api(monkeypatch, fake):
    stub = types.ModuleType("src.services.guara_identifier_service")
    stub.GuaraIdentifierService = lambda: fake
    monkeypatch.setitem(sys.modules, "src.services.guara_identifier_service", stub)
    monkeypatch.delitem(sys.modules, "src.api", raising=False)
    return importlib.import_module("src.api")


def _upload():
    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return UploadFile(
        file=io.BytesIO(buf.tobytes()),
        filename="teste.jpg",
        headers=Headers({"content-type": "image/jpeg"}),
    )


def test_inferencias_simultaneas_respeitam_o_limite(monkeypatch):
    fake = _FakeService()
    api = _load_api(monkeypatch, fake)
    app = api.create_app()
    endpoint = next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/guara-vermelho/inference")

    async def run():
        return await asyncio.gather(*(endpoint(_upload()) for _ in range(6)))

    results = asyncio.run(run())

    assert results == [{"results": []}] * 6
    assert fake.calls == 6
    assert fake.max_running == api.THREAD_POOL_WORKERS == 2
