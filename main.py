import argparse
import json

from src.services.guara_identifier_service import GuaraIdentifierService


class Application:
    def __init__(self) -> None:
        self._service = GuaraIdentifierService()

    def run(self, image_path: str) -> dict:
        return self._service.process_image(image_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identificacao de guaras em imagem")
    parser.add_argument("--image", required=True, help="Caminho da imagem para analise")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    app = Application()
    result = app.run(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()