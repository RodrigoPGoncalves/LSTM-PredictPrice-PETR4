import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime

class PreProcess:
    def __init__(self):
        self.symbol_usdt_to_brl = 'BRL=X'
        self.symbol_acao = 'PETR4.SA'
        self.start_date = '2014-02-22'
        self.end_date = '2025-05-05'
        self.start_date_test_model = '2025-05-05'
        self.end_date_test_model = '2025-05-12'
        self.path_files_original = "originalFiles"
        self.path_files_process = "preprocessFiles"

    def get_bacen_series(self, codigo_serie, data_inicio, data_fim):
        if data_fim is None:
            data_fim = datetime.today().strftime('%d/%m/%Y')

        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/"
            f"dados?formato=json&dataInicial={data_inicio}&dataFinal={data_fim}"
        )

        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Erro ao acessar API do Bacen: {response.status_code}")

        dados = response.json()
        df = pd.DataFrame(dados)
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df['valor'] = df['valor'].astype(float)
        return df.set_index('data')

        # Códigos SGS:
        # - Selic Meta ao mês: 432
        # - Selic Efetiva diária: 4189
        # - IPCA: 433
    
    def download_and_save_files(self):
        df_usdt_to_brl_origin = yf.download(self.symbol_usdt_to_brl, start=self.start_date, end=self.end_date)
        df_acao_origin = yf.download(self.symbol_acao, start=self.start_date, end=self.end_date)
        df_acao_origin_test_model = yf.download(self.symbol_acao, start=self.start_date_test_model, end=self.end_date_test_model)
        #selic_df = self.get_bacen_series(4189, self.start_date, self.end_date)  # Selic diária
        #ipca_df = self.get_bacen_series(433, self.start_date, self.end_date)    # IPCA mensal

        df_usdt_to_brl_origin.to_csv(self.path_files_original + "/usdt2brl.csv")
        df_acao_origin.to_csv(self.path_files_original + "/acao.csv")
        df_acao_origin_test_model.to_csv(self.path_files_original + "/acao_test_model.csv")
        #selic_df.to_csv(self.path_files_original + "/selic.csv")
        #ipca_df.to_csv(self.path_files_original + "/ipca.csv")
        
    def ajusting_tables_Pulp(self,df):
        df_a = df.copy()
        try:
            df_a['Data'] = pd.to_datetime(df_a['Data'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
        except:
            print("Erro na conversão de data")

        # Reverter para ordem crescente de data
        df_a = df_a.iloc[::-1].reset_index(drop=True)
        df_a['Data'] = pd.to_datetime(df_a['Data'], format='%Y-%m-%d')

        # Ajuste dos valores numéricos (troca '.' por nada e ',' por '.')
        for col in ['Último', 'Abertura', 'Máxima', 'Mínima']:
            df_a[col] = df_a[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

        # Selecionar só as colunas necessárias
        df_a = df_a[["Data", "Último", "Abertura", "Máxima", "Mínima"]]

        # Renomear as colunas
        df_a.rename(columns={"Data": "Date",
                            "Último": "close_Pulp",
                            "Máxima": "high_Pulp",
                            "Mínima": "low_Pulp",
                            "Abertura": "open_Pulp"},
                    inplace=True)

        # --- Parte nova: expandir para todos os dias entre a data mínima e máxima ---

        # Data mínima e máxima
        data_min = df_a['Date'].min()
        data_max = df_a['Date'].max()

        # Criar índice diário completo
        idx = pd.date_range(start=data_min, end=data_max, freq='D')

        # Reindexar para incluir todos os dias e preencher os valores faltantes com o último valor válido
        df_a = df_a.set_index('Date').reindex(idx).ffill().reset_index()

        # Renomear coluna 'index' para 'Date' novamente
        df_a.rename(columns={'index': 'Date'}, inplace=True)

        return df_a

    def ajusting_tables_dolar(self,df):
        df_a = df.copy()

        # Seleciona as colunas necessárias
        df_a = df_a[["Price","High","Low","Open","Close"]]

        # Remove as duas primeiras linhas (provavelmente cabeçalhos extras)
        df_a = df_a.drop([0, 1], axis=0).reset_index(drop=True)

        # Renomeia as colunas
        df_a.rename(columns={
            "Price": "Date",
            "Close": "close_dolar",
            "High": "high_dolar",
            "Low": "low_dolar",
            "Open": "open_dolar"
        }, inplace=True)

        # Converte coluna Date para datetime
        df_a['Date'] = pd.to_datetime(df_a['Date'], format='%Y-%m-%d')

        # Converte os valores para float
        for col in ["close_dolar","high_dolar","low_dolar","open_dolar"]:
            df_a[col] = df_a[col].astype(float)

        # --- Parte nova: expandir para todos os dias entre a data mínima e máxima ---
        data_min = df_a['Date'].min()
        data_max = df_a['Date'].max()

        idx = pd.date_range(start=data_min, end=data_max, freq='D')

        # Reindexar pelo índice diário, preenchendo valores faltantes com o último válido
        df_a = df_a.set_index('Date').reindex(idx).ffill().reset_index()

        # Renomear coluna 'index' para 'Date'
        df_a.rename(columns={'index': 'Date'}, inplace=True)

        return df_a

    def ajusting_tables_action(self,df):
        df_b = df

        df_b = df_b[["Price","High","Low","Open","Volume","Close"]]

        df_b = df_b.drop([0, 1], axis=0)

        df_b.rename(columns={"Price":"Date","Close": "close_acao", "High": "high_acao", "Low": "low_acao", "Open": "open_acao", "Volume": "volume_acao"}, inplace=True)
        df_b = df_b.reset_index(drop=True)

        for col in ["close_acao","high_acao","low_acao","open_acao","volume_acao"]:
            df_b[col] = df_b[col].astype(float)
        return df_b

    def ajusting_tables_indices_juros(self, df, info):
        df_b = df

        df_b['data'] = pd.to_datetime(df_b['data'])
        data_inicio = pd.to_datetime(self.start_date)
        data_fim = pd.to_datetime(self.end_date)
        
        datas_diarias = pd.date_range(start=data_inicio, end=data_fim, freq='D')

        df_diario = pd.DataFrame({'data': datas_diarias})
        df_diario = pd.merge_asof(df_diario.sort_values('data'),
                         df_b.sort_values('data'),
                         on='data',
                         direction='backward')

        df_diario.rename(columns={"data":"Date","valor":f"Value{info}"}, inplace=True)
        df_diario = df_diario.reset_index(drop=True)

        return df_diario

    def ajust_dates(self, df_a, df_b, df_c, df_d, df_e):
        # Padroniza as colunas 'Date' para datetime
        for df in [df_a, df_b, df_c, df_d, df_e]:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

        # Começa com as datas de df_b
        common_dates = set(df_b['Date'])
        
        # Intersecta as datas de todos os outros dfs com as datas de df_b
        for df in [df_a, df_c, df_d, df_e]:
            common_dates = common_dates.intersection(set(df['Date']))
        
        # Filtra todos os dataframes para ficarem só com as datas em common_dates
        df_a = df_a[df_a['Date'].isin(common_dates)].reset_index(drop=True)
        df_b = df_b[df_b['Date'].isin(common_dates)].reset_index(drop=True)
        df_c = df_c[df_c['Date'].isin(common_dates)].reset_index(drop=True)
        df_d = df_d[df_d['Date'].isin(common_dates)].reset_index(drop=True)
        df_e = df_e[df_e['Date'].isin(common_dates)].reset_index(drop=True)

        # Faz os merges sequenciais pela coluna 'Date'
        merged = df_b.copy()
        for df in [df_a, df_c, df_d, df_e]:
            merged = pd.merge(merged, df, on='Date', how='inner')

        # Remove a coluna 'Date' se não quiser manter
        merged = merged.drop(columns=['Date'])

        # Arredonda os valores numéricos para 3 casas decimais
        merged = merged.round(3)

        return merged


    
    def process_and_save_files(self):
        df_acao_origin = pd.read_csv(self.path_files_original + "/acao.csv")
        df_acao_origin_test_model_origin = pd.read_csv(self.path_files_original + "/acao_test_model.csv")
        df_usdt_to_brl_origin = pd.read_csv(self.path_files_original + "/usdt2brl.csv")
        df_pulp_origin = pd.read_csv(self.path_files_original + "/Dados Históricos - Petróleo Brent Futuros.csv")
        df_selic_origin = pd.read_csv(self.path_files_original + "/selic.csv")
        df_ipca_origin = pd.read_csv(self.path_files_original + "/ipca.csv")
        
        df_dolar = self.ajusting_tables_dolar(df_usdt_to_brl_origin)
        df_acao = self.ajusting_tables_action(df_acao_origin)
        df_acao_test_model = self.ajusting_tables_action(df_acao_origin_test_model_origin)
        df_pulp = self.ajusting_tables_Pulp(df_pulp_origin)
        df_selic = self.ajusting_tables_indices_juros(df_selic_origin, "_selic")
        df_ipca = self.ajusting_tables_indices_juros(df_ipca_origin, "_ipca")

        df_process = self.ajust_dates(df_dolar, df_acao, df_pulp, df_selic, df_ipca)
        
        df_dolar.to_csv(self.path_files_process + "/usdt2brl.csv", index=False)
        df_acao.to_csv(self.path_files_process + "/acao.csv", index=False)
        df_acao_test_model.to_csv(self.path_files_process + "/acao_test_model.csv", index=False)
        df_pulp.to_csv(self.path_files_process + "/pulp.csv", index=False)
        df_selic.to_csv(self.path_files_process + "/selic.csv", index=False)
        df_ipca.to_csv(self.path_files_process + "/ipca.csv", index=False)
        df_process.to_csv(self.path_files_process + "/resultado.csv", index=False)

        return df_process

    def return_test_values_action(self):
        return pd.read_csv(self.path_files_process + "/acao_test_model.csv")

