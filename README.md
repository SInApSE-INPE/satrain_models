# Estimativa e Detecção de Precipitação com XGBoost

Este algoritmo utiliza o benchmark **[SatRain](https://satrain.readthedocs.io/)**, que consiste em um conjunto de dados que combina observações de satélite e radar para o treinamento e a avaliação de modelos de Machine Learning.  
Para facilitar a implementação, também foi utilizada a biblioteca oficial [SatRain](https://github.com/ipwgml/satrain).

## Descrição
O código implementa um pipeline composto pelas seguintes etapas:
- **Pré-processamento** dos dados SatRain;
- **Treinamento** do modelo **XGBoost** para:
  - Estimativa de precipitação;
  - Detecção de precipitação;
- **Avaliação** do desempenho do modelo por meio de métricas estatísticas como: Bias, MSE, MAE, SMAPE, Coeficiente de correlação linear e Resolução efetiva.
  - Comparação do desempenho do XGBoost com os baselines: ERA5 e GPROF V7.
