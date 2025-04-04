#1. Função que lê os dados de um arquivo txt. A entrada é o nome com o caminho até o arquivo txt 
#   e a saída é uma lista onde cada item é uma linha do arquivo txt.

def le_dados(arq):
    vetor = []
    with open(arq,"r") as file:
        for linha in file:
            vetor.append(linha.strip())
    return vetor

arq = "Tests/nomes.txt"
v = le_dados(arq)
for linha in v:
    print(linha)