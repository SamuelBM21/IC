#3. Função que lê os dados de um arquivo csv. A entrada é o nome com o caminho até o 
#  arquivo csv e a saída é uma lista de listas onde cada item é uma linha do arquivo csv,
#  e cada item é uma lista de outros itens da coluna.

import pandas as pd

def le_csv(arq):
    df = pd.read_csv(arq)
    return df.values.tolist()

arq = "Tests/dados.csv"
lista = le_csv(arq)
print(lista)