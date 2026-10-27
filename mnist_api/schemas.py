"""Contratos JSON da API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EntradaPredicao(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    imagem: str = Field(
        min_length=1,
        max_length=8 * 1024 * 1024 + 64,
        description="Imagem de um único dígito em data URL PNG, JPEG ou WebP com base64.",
    )
    fundo: Literal["claro", "escuro"] = Field(
        default="claro",
        description="Cor do fundo da imagem original.",
    )


Probabilidade = Annotated[float, Field(ge=0, le=1)]


class SaidaPredicao(BaseModel):
    digito: int = Field(ge=0, le=9)
    confianca: float = Field(ge=0, le=1)
    probabilidades: list[Probabilidade] = Field(min_length=10, max_length=10)
    temperatura: float = Field(gt=0, description="Temperatura da calibração aplicada.")


class SaidaSaude(BaseModel):
    status: Literal["ok"]
    modelo: Literal["CNN MNIST"]
    modelo_carregado: bool
