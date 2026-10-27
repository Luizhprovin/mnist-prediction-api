"""Envie uma imagem local para a API em execução."""

import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

FORMATOS = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".webp": "image/webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulte a API com uma imagem de um dígito.")
    parser.add_argument("imagem", type=Path)
    parser.add_argument("--fundo", choices=("claro", "escuro"), default="claro")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()

    tipo = FORMATOS.get(args.imagem.suffix.lower())
    if tipo is None:
        parser.error("A imagem precisa ser PNG, JPEG ou WebP.")
    dados = base64.b64encode(args.imagem.read_bytes()).decode("ascii")
    entrada = {"imagem": f"data:{tipo};base64,{dados}", "fundo": args.fundo}
    requisicao = Request(
        args.url,
        data=json.dumps(entrada).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(requisicao, timeout=60) as resposta:
        print(json.dumps(json.load(resposta), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
