# API de Classificação de Dígitos MNIST

API REST que recebe uma imagem com **um único dígito manuscrito** e devolve a classe prevista e as probabilidades das dez classes. Ela disponibiliza para outras aplicações uma CNN treinada no [projeto MNIST anterior](https://github.com/Luizhprovin/mnist-digit-classification). Os pesos e os parâmetros necessários acompanham este repositório; usar a API não exige novo treinamento.

Um uso possível é transformar recortes de campos numéricos manuscritos em dados digitais para revisão e registro. A API cuida da classificação de cada dígito recortado; a localização de campos em uma página completa fica a cargo da aplicação que a consulta.

## Modelo e origem

A CNN foi treinada com imagens MNIST de 28 × 28 pixels. O [notebook do experimento original](https://github.com/Luizhprovin/mnist-digit-classification/blob/8197a4e/projeto.ipynb) documenta a divisão dos dados, o treinamento, a escolha do modelo e a avaliação. Na partição de teste reservada, a CNN acertou **13.859 de 14.000 imagens**, com **98,9929% de acurácia** e **F1 ponderado de 98,9933%**. As métricas registradas estão em [`reports/metricas_teste.csv`](reports/metricas_teste.csv).

As 70.000 imagens foram separadas, de forma estratificada e com semente 42, em **42.000 para treino**, **7.000 para validação**, **7.000 para calibração** e **14.000 para teste**. Para fotografias, o pré-processamento estima o fundo, destaca o traço, recorta o componente detectado, redimensiona para caber em 20 × 20 e centraliza em uma tela de 28 × 28. Os parâmetros foram definidos nas imagens de desenvolvimento do experimento anterior.

O modelo serializado está em `models/cnn.keras`. A temperatura de calibração das probabilidades (**1,512363**) está em `models/calibracao_cnn.json`; os parâmetros de preparação das fotografias estão em `config/preprocessamento.json`. A aplicação carrega os três arquivos uma vez na inicialização e mantém o modelo em memória durante as requisições. O pré-processamento de imagens foi reaproveitado do projeto original.

| Componente | Versão |
|---|---:|
| Python | 3.12 |
| FastAPI | 0.162.0 |
| Pydantic | 2.14.1 |
| Uvicorn | 0.54.0 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| Pillow | 12.3.0 |

As versões fixadas também constam de [`requirements.txt`](requirements.txt). A instalação exige acesso à internet; depois dela, a API funciona localmente sem baixar o MNIST.

## Instalação

Use **Python 3.12 de 64 bits**. Abra o terminal na pasta que contém este README e `requirements.txt`.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

**macOS ou Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Não é necessário ativar o ambiente virtual. Caso `py -3.12` ou `python3.12` não seja encontrado, instale essa versão do Python antes de continuar.

## Iniciar e consultar a API

No Windows:

```powershell
.\.venv\Scripts\python.exe -m uvicorn mnist_api.main:app --host 127.0.0.1 --port 8000
```

No macOS ou Linux:

```bash
.venv/bin/python -m uvicorn mnist_api.main:app --host 127.0.0.1 --port 8000
```

Aguarde a mensagem de inicialização completa. O servidor fica ativo enquanto esse terminal estiver aberto. Acesse [a documentação interativa](http://127.0.0.1:8000/docs) para ver os esquemas e testar as rotas, ou [o endpoint de saúde](http://127.0.0.1:8000/health) para confirmar que o modelo está carregado. Use `Ctrl+C` para encerrar. Se a porta 8000 estiver ocupada, substitua `--port 8000` por outra porta e ajuste o endereço usado nas consultas.

Com o servidor ativo, abra **outro terminal** na mesma pasta e consulte a imagem de exemplo, que contém um dígito 0 recortado de uma fotografia do experimento anterior:

```bash
# macOS ou Linux
.venv/bin/python examples/consultar.py examples/digito_0.png
```

```powershell
# Windows
.\.venv\Scripts\python.exe examples/consultar.py examples/digito_0.png
```

Para consultar outra imagem, troque o caminho do arquivo. O arquivo deve conter apenas um dígito recortado. Para traço claro sobre fundo escuro, acrescente `--fundo escuro`; se a porta foi alterada, acrescente `--url http://127.0.0.1:8001/predict` com a porta escolhida. O script de exemplo usa somente a biblioteca padrão do Python.

Também é possível testar pelo Swagger: abra **POST /predict → Try it out**, substitua o corpo pelo conteúdo completo de [`examples/requisicao_swagger.json`](examples/requisicao_swagger.json) e clique em **Execute**. O JSON contém uma imagem base64 válida de `examples/digito_0_swagger.png` e deve retornar o dígito 0.

## Contrato HTTP

`GET /health` responde com HTTP 200 quando o serviço está ativo e a CNN reside na memória:

```json
{"status": "ok", "modelo": "CNN MNIST", "modelo_carregado": true}
```

`POST /predict` recebe um objeto JSON. `imagem` é uma *data URL* com conteúdo base64 e tipo PNG, JPEG ou WebP; `fundo` aceita `"claro"` (padrão) ou `"escuro"`. O script `examples/consultar.py` cria esse JSON a partir de um arquivo local. O exemplo abaixo mostra o formato; o arquivo `examples/requisicao_swagger.json` contém um valor executável.

```json
{
  "imagem": "data:image/png;base64,<conteudo-base64-da-imagem>",
  "fundo": "claro"
}
```

Uma resposta HTTP 200 tem esta estrutura (valores ilustrativos da imagem de exemplo):

```json
{
  "digito": 0,
  "confianca": 0.9925475929064796,
  "probabilidades": [0.9925475929064796, 0.0000007576743388783258, 0.006335588706934351, 0.0000921773513891348, 0.000000379007384098319, 0.0000022410873660703378, 0.00007299033915998896, 0.000014636865650725106, 0.000027384751294302355, 0.0009062513100027916],
  "temperatura": 1.512362965069051
}
```

O vetor `probabilidades` está na ordem dos dígitos **0 a 9** e soma 1. `confianca` é a probabilidade calibrada da classe prevista. Os contratos de entrada e saída são definidos com Pydantic em `mnist_api/schemas.py`.

| Situação | HTTP |
|---|---:|
| Predição ou saúde válidas | 200 |
| JSON sem campos obrigatórios, tipo incorreto, fundo inválido ou campo extra | 422 |
| Base64 ilegível, formato incompatível ou imagem sem traço detectável | 400 |
| Falha inesperada durante a inferência | 500 |

A requisição aceita imagens codificadas de até **6 MiB** e até **12 milhões de pixels**. O tamanho máximo do campo `imagem` no JSON é 8 MiB. Se o modelo ou suas configurações estiverem ausentes ou inválidos, a inicialização falha em vez de oferecer uma API sem modelo.

## Verificação

Instale também a dependência de teste:

```bash
# macOS ou Linux
.venv/bin/python -m pip install -r requirements-dev.txt
TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python -m unittest discover -s tests -v
```

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes verificam o carregamento único na inicialização, `/health`, `/docs`, a previsão com a CNN real e os códigos HTTP para entradas inválidas e falhas internas. A imagem de exemplo foi retirada da foto de desenvolvimento `IMG_7057.png` do projeto original; ela serve para uma demonstração reproduzível, sem representar uma nova estimativa de desempenho.

## Solução de problemas

| Sintoma | Verificação |
|---|---|
| O comando de criação do ambiente não encontra Python 3.12 | Confirme a instalação da versão 3.12 de 64 bits e use o comando correspondente ao sistema operacional. |
| Falha ao importar TensorFlow no Windows por DLL ausente | Instale o [Microsoft Visual C++ Redistributable indicado pelo TensorFlow](https://www.tensorflow.org/install/pip#software_requirements) e tente novamente. |
| O servidor informa que um artefato está ausente | Confira `models/cnn.keras`, `models/calibracao_cnn.json` e `config/preprocessamento.json` na pasta clonada. |
| A porta 8000 está ocupada | Use outra porta no comando do Uvicorn e a mesma porta em `--url` no cliente. |
| A API devolve 400 para uma fotografia | Envie somente um dígito recortado, com traço visível, em PNG, JPEG ou WebP. |

## Organização

```text
mnist_api/
  main.py              aplicação e lifespan
  routers.py           rotas HTTP
  schemas.py           contratos Pydantic
  services.py          carregamento e inferência
  preprocessamento.py  preparação da imagem
models/                 CNN e calibração
config/                 parâmetros do pré-processamento
examples/               imagem e cliente de consulta
reports/                métricas do experimento original
tests/                  verificações da API
```

O resultado de 98,9929% foi medido em imagens do MNIST. Fotografias reais podem variar em iluminação, enquadramento e espessura do traço; o experimento anterior com fotos próprias foi pequeno e não isolou o efeito de cada fator. Por isso, a confiança devolvida pela API não garante acerto em imagens fora da distribuição de treinamento.
