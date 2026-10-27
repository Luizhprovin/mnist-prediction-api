"""Rotas HTTP da API."""

import logging

from fastapi import APIRouter, HTTPException, Request

from .schemas import EntradaPredicao, SaidaPredicao, SaidaSaude
from .services import ImagemInvalida

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=SaidaSaude)
def health(request: Request) -> SaidaSaude:
    servico = request.app.state.servico
    return SaidaSaude(status="ok", modelo="CNN MNIST", modelo_carregado=servico is not None)


@router.post("/predict", response_model=SaidaPredicao,
             responses={400: {"description": "Imagem inválida"},
                        500: {"description": "Falha de inferência"}})
def predict(entrada: EntradaPredicao, request: Request) -> SaidaPredicao:
    try:
        resultado = request.app.state.servico.predizer(entrada.imagem, entrada.fundo)
    except ImagemInvalida as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Falha na inferência da CNN")
        raise HTTPException(status_code=500, detail="Não foi possível executar a predição.") from exc
    try:
        return SaidaPredicao.model_validate(resultado)
    except Exception as exc:
        logger.exception("Resposta inválida do serviço de inferência")
        raise HTTPException(status_code=500, detail="Não foi possível executar a predição.") from exc
