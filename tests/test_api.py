"""Contrato HTTP e inferência real com a CNN entregue."""

import base64
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from mnist_api.main import app, criar_app
from mnist_api.services import ServicoPredicao

AMOSTRA = Path(__file__).resolve().parents[1] / "examples/digito_0.png"


def data_url(dados: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(dados).decode("ascii")


class TesteAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.amostra = data_url(AMOSTRA.read_bytes())
        cls.patch_carregamento = patch.object(
            ServicoPredicao, "carregar", wraps=ServicoPredicao.carregar,
        )
        cls.carregamento = cls.patch_carregamento.start()
        cls.contexto = TestClient(app)
        try:
            cls.cliente = cls.contexto.__enter__()
        except Exception:
            cls.patch_carregamento.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.contexto.__exit__(None, None, None)
        cls.patch_carregamento.stop()

    def test_health_e_documentacao(self):
        resposta = self.cliente.get("/health")
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.json(), {
            "status": "ok", "modelo": "CNN MNIST", "modelo_carregado": True,
        })
        self.assertEqual(self.cliente.get("/docs").status_code, 200)
        esquema = self.cliente.get("/openapi.json").json()
        self.assertIn("/predict", esquema["paths"])
        self.assertIn("/health", esquema["paths"])

    def test_predicao_real_e_carregamento_unico(self):
        for _ in range(2):
            resposta = self.cliente.post(
                "/predict", json={"imagem": self.amostra, "fundo": "claro"},
            )
            self.assertEqual(resposta.status_code, 200)
            resultado = resposta.json()
            self.assertEqual(resultado["digito"], 0)
            self.assertEqual(len(resultado["probabilidades"]), 10)
            self.assertAlmostEqual(sum(resultado["probabilidades"]), 1, places=8)
            self.assertAlmostEqual(
                resultado["confianca"], resultado["probabilidades"][0], places=8,
            )
            self.assertAlmostEqual(resultado["confianca"], 0.992548, places=4)
            self.assertAlmostEqual(resultado["temperatura"], 1.512363, places=5)
        self.assertEqual(self.carregamento.call_count, 1)

    def test_traco_claro_em_fundo_escuro(self):
        with Image.open(AMOSTRA) as original:
            imagem = ImageOps.invert(original.convert("RGB"))
        memoria = BytesIO()
        imagem.save(memoria, format="PNG")
        resposta = self.cliente.post(
            "/predict", json={"imagem": data_url(memoria.getvalue()), "fundo": "escuro"},
        )
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.json()["digito"], 0)

    def test_entrada_invalida_retorna_422(self):
        casos = [
            {},
            {"imagem": 123},
            {"imagem": self.amostra, "fundo": "azul"},
            {"imagem": self.amostra, "extra": True},
        ]
        for caso in casos:
            with self.subTest(caso=caso.keys()):
                self.assertEqual(self.cliente.post("/predict", json=caso).status_code, 422)

    def test_imagem_invalida_retorna_400(self):
        branco = Image.new("RGB", (64, 64), "white")
        memoria = BytesIO()
        branco.save(memoria, format="PNG")
        casos = [
            "data:image/png;base64,@@@@@@@@@@@@@@",
            data_url(b"conteudo que nao e uma imagem"),
            data_url(memoria.getvalue()),
        ]
        for imagem in casos:
            with self.subTest(imagem=imagem[:45]):
                resposta = self.cliente.post("/predict", json={"imagem": imagem})
                self.assertEqual(resposta.status_code, 400)
                self.assertIn("detail", resposta.json())

    def test_falha_interna_retorna_500_sem_expor_detalhes(self):
        with patch.object(app.state.servico, "predizer", side_effect=RuntimeError("segredo")):
            with patch("mnist_api.routers.logger.exception"):
                resposta = self.cliente.post("/predict", json={"imagem": self.amostra})
        self.assertEqual(resposta.status_code, 500)
        self.assertNotIn("segredo", resposta.text)


class TesteInicializacao(unittest.TestCase):
    def test_artefatos_ausentes_impedem_inicio(self):
        with TemporaryDirectory() as pasta:
            with patch("mnist_api.main.RAIZ", Path(pasta)):
                with self.assertRaises(FileNotFoundError):
                    with TestClient(criar_app()):
                        pass


if __name__ == "__main__":
    unittest.main()
