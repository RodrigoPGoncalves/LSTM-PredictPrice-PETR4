import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/"

def grafico_predict(json_value):
    df_acao_test_model = pd.DataFrame(json_value['df_acao_test_model'])
    df_acao_test_model['Date'] = pd.to_datetime(df_acao_test_model['Date'])

    date_predict = pd.to_datetime(json_value['date_predict'])

    # Criando o gráfico interativo
    fig = go.Figure()

    # Adicionando os dados históricos
    fig.add_trace(go.Scatter(
        x=df_acao_test_model['Date'],
        y=df_acao_test_model['close_acao'],
        mode='lines+markers',
        name='Dados Históricos',
        line=dict(color='blue')
    ))

    # Adicionando previsões
    fig.add_trace(go.Scatter(
        x=date_predict,
        y=json_value['predict_sample_lstm'],
        mode='lines+markers',
        name='Previsão LSTM',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=date_predict,
        y=json_value['predict_lstm_bidirecional'],
        mode='lines+markers',
        name='Previsão LSTM Bidirecional',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=date_predict,
        y=json_value['predict_lstm_attention'],
        mode='lines+markers',
        name='Previsão LSTM + Attention',
        line=dict(color='gray')
    ))

    fig.add_trace(go.Scatter(
        x=date_predict,
        y=json_value['predict_lstm_cnn'],
        mode='lines+markers',
        name='Previsão LSTM + CNN',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=date_predict,
        y=json_value['predict_lstm_bi_atten_cnn'],
        mode='lines+markers',
        name='Previsão LSTM Bi-Attention + CNN',
        line=dict(color='pink')
    ))

    # Customizando o layout
    fig.update_layout(
        title='Previsões de Preço de Fechamento',
        xaxis_title='Datas',
        yaxis_title='Preço de Fechamento',
        legend_title='Modelos',
        template='plotly_white'
    )

    # Exibindo o gráfico no Streamlit
    st.plotly_chart(fig)

def app():
    st.write("""
            Abaixo será mostrado o trabalho de substituição da Fase 5 - Deep Learning e IA, que tem o intuito de realizar previsões do valor de
             fechamento da ação acao da petrobras PETR4, cuja a fonte principal é a comodite petroleo. 
             Os modelos treinados consideram a data inicial de treinamento como 2014-02-22 e a final 2025-05-04, logo as previsões serão dadas
             a partir do dia 2025-05-05, fiz desta forma para possuir um bom range de inserção de dados e comparação com o comportamento
             real da ação.
            """)
    st.write("""Para o treinamento, foram utilizados os dados da ação acao do dólar em comparação ao real, o brent do petroleo futuro e dados de inflação e selic
              no Brasil, cujo os links de acesso estão abaixo:
             """)
    st.markdown("[acao](https://finance.yahoo.com/quote/PETR4.SA/)", unsafe_allow_html=True)
    st.markdown("[Dólar](https://finance.yahoo.com/quote/BRL=X/)", unsafe_allow_html=True)
    st.markdown("[Petróleo Brent Futuros ](https://br.investing.com/commodities/brent-oil-historical-data)", unsafe_allow_html=True)
    st.markdown("[IPCA e Selic](https://api.bcb.gov.br/dados/serie)", unsafe_allow_html=True)
    st.write("""Para escolher os modelos aqui já pré-treinados eu utilizei o mlflow para realizar log de métricas e gráficos para cada modelo,
             desta forma ficou muito mais fácil de comparar e armazenar as informações, pois utilizei de técnicas de randomização
             para busca de hiperparâmetros (No arquivo de mlflow tem menos modelos, pois exclui alguns para diminuir o tamanho do arquivo).
             """)
    st.write("LEMBRETE: QUE AS AÇÕES (IBOVESPA) NÃO FUNCIONA AS DOMINGOS E FERIADOS")

    st.header("Carregue os modelos antes de iniciar: ")
    if st.button("Carregar Modelos"):
        with st.spinner("Carregando, aguarde..."):
            try:
                response = requests.get(
                    API_URL + "loadModels",
                    timeout=120 
                )

                if response.status_code == 200:
                    data = response.json()
                    st.success("Modelos carregados com sucesso!")
                else:
                    st.error(f"Erro na API: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Erro ao conectar com a API: {str(e)}")

    
    tab1, tab2, tab3, tab4 = st.tabs(["Previsão por período", "Previsão com valores inseridos", "Modelos Utilizados", "Melhorias e conclusão"])
    with tab1:
        st.header("Previsão dado período desejado: ")
        numb_days_predict = st.number_input(label="Período em dias:",key="input1", min_value=0, max_value=120, step=1)
        if st.button("Prever"):
            if numb_days_predict > 0:
                with st.spinner("Processando, aguarde..."):
                    try:
                        response = requests.post(
                            API_URL + "predict_period",
                            json={"number": numb_days_predict},
                            timeout=60 
                        )

                        if response.status_code == 200:
                            data = response.json()
                            st.success("Previsão concluída com sucesso!")
                            grafico_predict(data["data"]) 
                        else:
                            st.error(f"Erro na API: {response.status_code} - {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Erro ao conectar com a API: {str(e)}")
            else:
                st.error("Por favor, insira um número válido maior que zero.")
    with tab2:
        st.header("Previsão inserindo valores: ")
        st.write("""Para cada categoria abaixo é necessário inserir um valor (mesmo que repetido se desejar), tais 
                valores serão utilizados, não para retreinar, mas para ajudar na previsão de dados mais recentes, como falei o 
                primeiro dado inserido será considerado na base de predição como o dia 2025-05-05 e assim subsequente, logo, com 
                dados mais recentes inseridos, haverá melhores predições para o período especificado.
                """)
        columns = [
        'high_dolar', 'low_dolar', 'open_dolar', 'close_dolar', 'close_TIOc1',
        'open_TIOc1', 'high_TIOc1', 'low_TIOc1', 'high_acao', 'low_acao',
        'open_acao', 'volume_acao', 'close_acao']

        if "data" not in st.session_state:
            st.session_state.data = []

        new_entry = {}
        for column in columns:
            new_entry[column] = st.text_input(f"Insira o valor para {column}", value="", key=column)

        if st.button("Adicionar"):
            try:
                converted_entry = {k: float(v) for k, v in new_entry.items()}
                st.session_state.data.append(converted_entry)
                st.success("Entrada adicionada com sucesso!")
            except ValueError:
                st.error("Certifique-se de que todos os valores sejam numéricos (inteiros ou flutuantes com ponto).")

        if st.session_state.data:
            st.subheader("Dados adicionados")
            df = pd.DataFrame(st.session_state.data)
            st.dataframe(df)

        numb_days_predict_new_values = st.number_input(label="Período em dias:", key="input2", min_value=0, max_value=120, step=1)

        if st.button("Run"):
            if numb_days_predict_new_values > 0:
                if st.session_state.data:
                    with st.spinner("Inserindo dados e calculando predição, aguarde..."):
                        try:
                            df = pd.DataFrame(st.session_state.data)
                            payload = df.to_dict(orient="records")
                            
                            response = requests.post(
                                API_URL + "predict_new_values_period",
                                json={"number": numb_days_predict_new_values,
                                    "data": payload},
                                timeout=300 
                            )
                            if response.status_code == 200:
                                data = response.json()
                                st.success("Previsão concluída com sucesso!")
                                grafico_predict(data["data"])
                            else:
                                st.error(f"Erro na API: {response.status_code} - {response.text}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Erro ao conectar com a API: {str(e)}")
                else:
                    st.warning("Nenhum dado foi adicionado.")
            else:
                st.error("Por favor, insira um número válido maior que zero.")
    with tab3:
        st.title("Modelos Utilizados para Predição")

        st.write("""
        A seguir, são apresentados os tipos de LSTM (Long Short-Term Memory) que foram utilizados 
        para as predições no sistema. Cada modelo possui características específicas para lidar com diferentes aspectos da sequência de dados.
        """)

        st.header("Modelo 1: LSTM Pura")
        st.write("""
        Este é o modelo básico de LSTM. Ele utiliza camadas padrão de LSTM para capturar dependências de longo prazo nos dados sequenciais. 
        É ideal para séries temporais e tarefas que exigem a preservação do contexto ao longo do tempo.""")
        data1 = {
            "batch_size": [128],
            "dropout_rate": [0.2],
            "dropout_rate2": [0.2],
            "dropout_rate3": [0.4],
            "epochs": [30],
            "hidden_units_1": [64],
            "hidden_units_2": [512],
            "hidden_units_3": [32],
            "sequence_length": [5],
            "MAE": [0.042],
            "MAPE": [6.45],
            "MSE": [0.0025],
            "RMSE": [0.05],
            "R²": [0.9488],
            "loss": [0.014],
        }
        df1 = pd.DataFrame(data1)
        st.subheader("Tabela de melhores métricas")
        st.dataframe(df1)
        
        st.header("Modelo 2: LSTM Bidirecional") 
        st.write("""
        A LSTM bidirecional processa a sequência de dados em duas direções: 
        para frente (do início ao fim) e para trás (do fim ao início). 
        Isso permite que o modelo capture dependências passadas e futuras, 
        tornando-o mais robusto para certas tarefas.""")
        
        data2 = {
            "batch_size": [32],
            "dropout_rate": [0.3],
            "dropout_rate2": [0.2],
            "dropout_rate3": [0.3],
            "epochs": [100],
            "hidden_units_1": [128],
            "hidden_units_2": [512],
            "hidden_units_3": [64],
            "sequence_length": [3],
            "MAE": [0.024],
            "MAPE": [4.032],
            "MSE": [0.00089],
            "RMSE": [0.0298],
            "R²": [0.9818],
            "loss": [0.0037],
        }
        df2 = pd.DataFrame(data2)
        st.subheader("Tabela de melhores métricas")
        st.dataframe(df2)

        st.header("Modelo 3: LSTM com Attention")
        st.write("""
        Este modelo integra um mecanismo de Attention à LSTM. 
        O mecanismo de Attention ajuda o modelo a identificar quais partes da sequência são mais relevantes 
        para a predição atual, permitindo um foco mais preciso em informações críticas.""")
        
        data3 = {
            "batch_size": [64],
            "dropout_rate": [0.2],
            "dropout_rate2": [0.2],
            "dropout_rate3": [0.2],
            "epochs": [30],
            "hidden_units_1": [32],
            "hidden_units_2": [256],
            "hidden_units_3": [128],
            "sequence_length": [3],
            "MAE": [0.045],
            "MAPE": [7.866],
            "MSE": [0.002],
            "RMSE": [0.0511],
            "R²": [0.953],
            "loss": [0.0061],
        }

        df3 = pd.DataFrame(data3)
        st.subheader("Tabela de melhores métricas")
        st.dataframe(df3)

        st.header("Modelo 4: LSTM CNN")
        st.write("""
        Este modelo combina LSTM com camadas Convolucionais (CNN). 
        As CNNs extraem padrões locais significativos dos dados, enquanto a LSTM captura as dependências temporais. 
        Essa combinação é especialmente útil em séries temporais com características locais relevantes.""")
        
        data4 = {
            "batch_size": [128],
            "dropout_rate": [0.2],
            "dropout_rate2": [0.3],
            "dropout_rate3": [0.4],
            "epochs": [50],
            "hidden_units_1": [256],
            "hidden_units_2": [128],
            "hidden_units_3": [256],
            "sequence_length": [12],
            "MAE": [0.037],
            "MAPE": [5.095],
            "MSE": [0.0021],
            "RMSE": [0.0458],
            "R²": [0.9357],
            "loss": [0.012],
        }

        df4 = pd.DataFrame(data4)
        st.subheader("Tabela de melhores métricas")
        st.dataframe(df4)

        st.header("Modelo 5: LSTM Bidirecional + Attention + CNN")
        st.write("""
        Este é o modelo mais avançado utilizado, combinando LSTM bidirecional, 
        mecanismo de Attention e camadas CNN. Ele aproveita o melhor de cada técnica: 
        dependências passadas e futuras (bidirecional), foco em informações relevantes (Attention) 
        e extração de padrões locais (CNN). 
        É uma abordagem poderosa para problemas complexos de séries temporais.""")

        data5 = {
            "batch_size": [128],
            "dropout_rate": [0.4],
            "dropout_rate2": [0.4],
            "dropout_rate3": [0.2],
            "epochs": [50],
            "hidden_units_1": [256],
            "hidden_units_2": [128],
            "hidden_units_3": [256],
            "sequence_length": [18],
            "MAE": [0.1],
            "MAPE": [14.11],
            "MSE": [0.014],
            "RMSE": [0.118],
            "R²": [0.75],
            "loss": [0.014],
        }

        df5 = pd.DataFrame(data5)
        st.subheader("Tabela de melhores métricas")
        st.dataframe(df5)
    with tab4:
        st.write("""
                    Os melhores modelos, disparados, quanto em relação a proximidade aos valores previstos e tendências futuras é o modelo 2 e 4, acredito
                 que isso aconteça pela utilização da CNN, que são muito boas em encontrar características profundas nos dados e o bidirecional por se tratar
                 de um modelo que captura informações futuras e passadas. Acredito também que o modelo 5 que envolve a junção de várias técnicas ao mesmo tempo
                 não se sobressaiu perante outros por falta de melhores ajustes nos hiperparametros. 
                    Algo que eu gostaria de ter testado era utilizar a rede pré treinada do VGG16 e ai no final dela inserir uma LSTM sem contar
                 que analises de susbstiuição de valores, como nos finais de semana ou feriados,apesar disso foram copiados os valores anteriores 
                 (index-1) para os dados de dolar, brent selic e ipca para garantir que houvesssem mais dados para o treinamento, talvez a utilização da 
                 média ou mediana ajudem a melhorar os valores de previsão.
                """)


if __name__ == "__main__":
    app()