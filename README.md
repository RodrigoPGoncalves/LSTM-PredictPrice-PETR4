# 📈 LSTM-PETR4 - Previsão de Ações com Deep Learning

Este projeto tem como objetivo prever o comportamento da ação PETR4 utilizando modelos de deep learning como LSTM, CNN-LSTM, LSTM com Attention e outros. Ele inclui pipelines de pré-processamento, treinamento e testes, além de uma interface via Streamlit e uma API de comunicação com o back.

---

## 📁 Estrutura do Projeto
```
├── models/ # Modelos de deep learning
│ └── modelsTrainLocal/ # Implementações específicas (LSTM, CNN, etc.)
│
├── preprocess/ # Scripts de pré-processamento
├── routes/ # API (FastAPI ou Flask)
├── streamlitPages/ # Interface em Streamlit
├── only_testes/ # Notebooks e scripts de validação
├── originalFiles/ # Arquivos de dados originais (csv)
├── preprocessFiles/ # Arquivos de dados já pré-processados
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo
```


---

## 🚀 Rodando Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/LSTM-PETR4.git
cd LSTM-PETR4
```

### 2. Criar e ativar ambiente virtual (opcional, recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Rodar o Programa

```bash
python __init__.py
```
### 5. Acessos


- Fast API em: http://127.0.0.1:8000/

- Streamlit em: http://127.0.0.1:8501/

- Mlflow em: http://127.0.0.1:5000/

## 🐳 Rodando com Docker

1. Construir e subir os containers

docker-compose up --build

## 📊 Modelos disponíveis
- model_lstm.py


- model_lstm_cnn.py


- model_lstm_attention.py


- model_lstm_bidirecional.py


- model_lstm_bi_atten_cnn.py

## 📎 Dados
Os arquivos .csv utilizados estão na pasta originalFiles/, incluindo dados de:


- PETR4


- SELIC


- IPCA


- Dólar (USDT-BRL)


- Petróleo Brent

## 📦 Requisitos


- Python 3.8+


- Pandas, NumPy, TensorFlow, Keras


- Streamlit


- FastAPI 