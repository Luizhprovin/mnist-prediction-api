"""Carregamento e inferência do modelo, independentes do protocolo HTTP."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
import json
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.special import softmax

from .preprocessamento import preparar_digito

FORMATOS = {
    "data:image/png;base64": "PNG",
    "data:image/jpeg;base64": "JPEG",
    "data:image/webp;base64": "WEBP",
}
LIMITE_BYTES_IMAGEM = 6 * 1024 * 1024
LIMITE_PIXELS = 12_000_000


class ImagemInvalida(ValueError):
    """Entrada reconhecida pelo contrato JSON, mas sem imagem utilizável."""


def decodificar_imagem(data_url: str) -> Image.Image:
    """Valide e decodifique a imagem enviada na requisição."""
    cabecalho, separador, conteudo = data_url.partition(",")
    if not separador or cabecalho not in FORMATOS:
        raise ValueError("Envie uma imagem PNG, JPEG ou WebP em data URL com base64.")
    try:
        dados = base64.b64decode(conteudo, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("O conteúdo base64 da imagem é inválido.") from exc
    if not dados or len(dados) > LIMITE_BYTES_IMAGEM:
        raise ValueError("A imagem está vazia ou ultrapassa 6 MiB.")
    try:
        with Image.open(BytesIO(dados)) as imagem:
            if imagem.format != FORMATOS[cabecalho]:
                raise ValueError("O tipo da imagem não corresponde ao conteúdo enviado.")
            if imagem.width * imagem.height > LIMITE_PIXELS:
                raise ValueError("A imagem ultrapassa 12 milhões de pixels.")
            imagem.load()
            return imagem.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Não foi possível ler a imagem enviada.") from exc


def ajustar_cores(imagem: Image.Image, fundo: str) -> Image.Image:
    """Aplique orientação EXIF e prepare traço escuro sobre fundo claro."""
    imagem = ImageOps.exif_transpose(imagem)
    if "A" in imagem.getbands() or "transparency" in imagem.info:
        rgba = imagem.convert("RGBA")
        base = Image.new("RGBA", rgba.size, "white" if fundo == "claro" else "black")
        imagem = Image.alpha_composite(base, rgba)
    rgb = imagem.convert("RGB")
    return ImageOps.invert(rgb) if fundo == "escuro" else rgb


class ServicoPredicao:
    """Mantenha a CNN e os parâmetros em memória entre as requisições."""

    def __init__(self, extrator, pesos: np.ndarray, vies: np.ndarray,
                 temperatura: float, parametros: dict):
        self._extrator = extrator
        self._pesos = pesos
        self._vies = vies
        self.temperatura = temperatura
        self.parametros = parametros
        self._lock = Lock()

    @classmethod
    def carregar(cls, modelo_path: Path, calibracao_path: Path,
                preprocessamento_path: Path) -> ServicoPredicao:
        """Leia os três artefatos apenas na inicialização da aplicação."""
        calibracao = json.loads(calibracao_path.read_text(encoding="utf-8"))
        processamento = json.loads(preprocessamento_path.read_text(encoding="utf-8"))
        temperatura = float(calibracao["temperatura"])
        if (calibracao.get("metodo") != "temperature_scaling"
                or not np.isfinite(temperatura) or temperatura <= 0
                or processamento.get("versao") != "1.0"):
            raise ValueError("Os artefatos de calibração ou pré-processamento são inválidos.")

        import tensorflow as tf

        modelo = tf.keras.models.load_model(modelo_path, compile=False)
        if tuple(modelo.input_shape[1:]) != (28, 28, 1) or modelo.output_shape[-1] != 10:
            raise ValueError("A CNN precisa receber 28 × 28 × 1 e produzir dez classes.")
        pesos, vies = modelo.layers[-1].get_weights()
        extrator = tf.keras.Model(inputs=modelo.inputs, outputs=modelo.layers[-1].input)
        return cls(
            extrator=extrator,
            pesos=pesos.astype(np.float64),
            vies=vies.astype(np.float64),
            temperatura=temperatura,
            parametros=processamento["parametros"],
        )

    def predizer(self, data_url: str, fundo: str) -> dict:
        try:
            imagem = ajustar_cores(decodificar_imagem(data_url), fundo)
            entrada = preparar_digito(imagem, **self.parametros)[None, ..., None]
        except ValueError as exc:
            raise ImagemInvalida(str(exc)) from exc
        with self._lock:
            caracteristicas = self._extrator.predict(entrada, verbose=0)
        logits = caracteristicas.astype(np.float64) @ self._pesos + self._vies
        probabilidades = softmax(logits / self.temperatura, axis=1)[0]
        if not np.isfinite(probabilidades).all() or not np.isclose(probabilidades.sum(), 1):
            raise RuntimeError("O modelo não retornou probabilidades válidas.")
        digito = int(probabilidades.argmax())
        return {
            "digito": digito,
            "confianca": float(probabilidades[digito]),
            "probabilidades": [float(p) for p in probabilidades],
            "temperatura": self.temperatura,
        }
