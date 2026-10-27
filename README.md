# API de Classificação de Dígitos MNIST

Serviço de inferência para reconhecer um dígito manuscrito em uma imagem. O modelo é a CNN treinada e calibrada no [projeto MNIST anterior](https://github.com/Luizhprovin/mnist-digit-classification), preservado neste repositório como origem do experimento.

A CNN alcançou **98,9929% de acurácia** (13.859 acertos em 14.000 imagens) no conjunto de teste reservado. O resultado completo está em [`reports/metricas_teste.csv`](reports/metricas_teste.csv). O treinamento, a divisão dos dados e a avaliação podem ser consultados no [notebook original](https://github.com/Luizhprovin/mnist-digit-classification/blob/8197a4e/projeto.ipynb).

O arquivo `models/cnn.keras` contém os pesos treinados. `models/calibracao_cnn.json` guarda a temperatura usada para calibrar as probabilidades; `config/preprocessamento.json` registra os parâmetros de preparação das imagens. Esses artefatos acompanham este projeto para permitir a execução da API sem novo treinamento.
