#2. Função que escreve os dados de uma lista de strings em um arquivo txt.
#   A entrada é o nome com o caminho até o arquivo txt além da lista de strings.
#   Não há saída. Cada linha do arquivo é um item da lista.

def escreve_dados(arq, strings):
    with open(arq,"w") as file:
        for linha in strings:
            file.write(linha + "\n")

strings = ["Samuel","Curry", "Messi", "Lebron"]
arq = "Tests/escrever.txt"

escreve_dados(arq,strings)