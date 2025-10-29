#4. Função que escreve os dados de uma lista de listas de strings em um arquivo csv.
#   A entrada é o nome com o caminho até o arquivo csv e a lista de listas de strings e não há saída. 
#   Cada linha do arquivo é um item da lista, e cada coluna é um item dessa lista.

import pandas as pd
def escreve_csv(arq,lista):
    df = pd.DataFrame(lista)
    df.to_csv(arq, index=False, header=False)
    

lista = [
    ["Samuel", "18", "Itabirito"],
    ["Esther", "18", "Ouro Preto"],
    ["Thomaz", "11", "Belo Horizonte"]
]
arq = "Tests/escreve_csv.csv"

escreve_csv(arq,lista)