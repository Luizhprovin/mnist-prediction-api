"""Processamento das fotografias, compartilhado com o notebook.

Os valores padrão são os da versão 1.0, definida nas dez imagens de
desenvolvimento. As etapas expostas são as mesmas usadas na entrada da CNN.
"""

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, center_of_mass, gaussian_filter, label, shift


def etapas_digito(
    recorte: Image.Image,
    limiar_contraste: float = 0.15,
    tamanho_interno: int = 20,
    expansao_relativa: float = 0.02,
) -> dict[str, np.ndarray]:
    """Prepare um traço escuro sobre fundo claro e exponha as transformações.

    Não estima novos parâmetros nem altera o modelo. Uma entrada uniforme ou
    sem componente de tamanho suficiente é recusada antes da inferência.
    """
    if min(recorte.size) < 2:
        raise ValueError("A imagem precisa ter pelo menos 2 × 2 pixels.")
    if not 0 < limiar_contraste < 1:
        raise ValueError("O limiar de contraste deve estar entre 0 e 1.")
    if not isinstance(tamanho_interno, int) or not 1 <= tamanho_interno <= 26:
        raise ValueError("O tamanho interno deve ser um inteiro de 1 a 26.")
    if not 0 < expansao_relativa <= 0.1:
        raise ValueError("A expansão relativa deve estar entre 0 e 0,1.")

    cinza = np.asarray(ImageOps.grayscale(recorte), dtype=np.float32) / 255.0
    fundo = gaussian_filter(cinza, sigma=max(cinza.shape) / 15)
    contraste = np.clip((fundo - cinza) / np.maximum(fundo, 1e-6), 0, 1)
    mascara = contraste >= limiar_contraste
    componentes, quantidade = label(mascara)
    if quantidade == 0:
        raise ValueError("Nenhum traço encontrado. Desenhe ou recorte um dígito.")
    tamanhos = np.bincount(componentes.ravel())
    tamanhos[0] = 0
    mascara = tamanhos[componentes] >= max(3, tamanhos.max() * 0.01)
    ys, xs = np.nonzero(mascara)
    if len(ys) == 0:
        raise ValueError("O traço é muito pequeno. Use um dígito mais visível.")
    digito = mascara[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    expansao = max(1, round(max(digito.shape) * expansao_relativa))
    digito = binary_dilation(np.pad(digito, expansao + 1), iterations=expansao)
    ys, xs = np.nonzero(digito)
    digito = digito[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    altura, largura = digito.shape
    escala = tamanho_interno / max(altura, largura)
    tamanho = (max(1, round(largura * escala)), max(1, round(altura * escala)))
    reduzido = Image.fromarray(digito.astype(np.uint8) * 255).resize(
        tamanho, Image.Resampling.LANCZOS,
    )
    tela = np.zeros((28, 28), dtype=np.float32)
    x0, y0 = (28 - tamanho[0]) // 2, (28 - tamanho[1]) // 2
    tela[y0:y0 + tamanho[1], x0:x0 + tamanho[0]] = (
        np.asarray(reduzido, dtype=np.float32) / 255.0
    )
    centro_y, centro_x = center_of_mass(tela)
    tela = shift(
        tela, (13.5 - centro_y, 13.5 - centro_x),
        order=1, mode="constant", cval=0, prefilter=False,
    )
    tela = np.clip(tela, 0, 1).astype(np.float32)
    if tela.shape != (28, 28) or not np.isfinite(tela).all() or tela.max() <= 0:
        raise ValueError("Não foi possível preparar uma entrada válida para a CNN.")
    return {
        "cinza": cinza,
        "fundo": fundo,
        "contraste": contraste,
        "traco": mascara.astype(np.float32),
        "entrada": tela,
    }


def preparar_digito(
    recorte: Image.Image,
    limiar_contraste: float = 0.15,
    tamanho_interno: int = 20,
    expansao_relativa: float = 0.02,
) -> np.ndarray:
    """Retorne a entrada 28 × 28, float32 em [0, 1], usada pelo notebook."""
    return etapas_digito(
        recorte, limiar_contraste, tamanho_interno, expansao_relativa,
    )["entrada"]
