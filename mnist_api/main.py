"""Aplicação FastAPI da CNN MNIST."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .routers import router
from .services import ServicoPredicao

RAIZ = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.servico = ServicoPredicao.carregar(
        RAIZ / "models/cnn.keras",
        RAIZ / "models/calibracao_cnn.json",
        RAIZ / "config/preprocessamento.json",
    )
    try:
        yield
    finally:
        app.state.servico = None


def criar_app() -> FastAPI:
    aplicacao = FastAPI(
        title="API de Classificação de Dígitos MNIST",
        description="Inferência de um dígito manuscrito por imagem com CNN e probabilidades calibradas.",
        version="1.0.0",
        lifespan=lifespan,
    )
    aplicacao.include_router(router)
    return aplicacao


app = criar_app()
