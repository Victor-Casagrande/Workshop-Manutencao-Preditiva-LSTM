# Manutenção Preditiva com Deep Learning (LSTM)

Este repositório contém o material prático do Workshop **"Da Teoria à Prática"**, desenvolvido como Projeto de Extensão do curso de Ciência da Computação (IFC - Campus Videira).

## Objetivo
Demonstrar como aplicar Inteligência Artificial (Redes Neurais LSTM) para prever falhas em motores turbofan, estimando a Vida Útil Remanescente (RUL).

## Tecnologias Usadas
* **Python 3.13**
* **TensorFlow/Keras** (Deep Learning)
* **Pandas & NumPy** (Manipulação de Dados)
* **Matplotlib** (Visualização)

## Estrutura do Projeto
* `Workshop_RUL_Final.ipynb`: O guia interativo passo-a-passo (Jupyter Notebook).
* `CMAPSSData/`: Pasta contendo os datasets da NASA (FD001).
* `modelo_rul_lstm.h5`: O modelo de IA já treinado.

## Como Executar
1. Clone este repositório.
2. Instale as dependências: `pip install pandas numpy matplotlib tensorflow scikit-learn`
3. Abra o arquivo `.ipynb` no VS Code ou Google Colab.
4. Execute as células sequencialmente.

## Resultados
O modelo atingiu um **RMSE de ~15 ciclos** no conjunto de teste, demonstrando capacidade de generalização na previsão de falhas.

---
*Autor: Victor Casagrande*