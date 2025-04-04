#5. Função que recebe o caminho até uma pasta e retorna uma lista
#   com todos os arquivos dentro dessa pasta e suas subpastas.


from os import listdir #https://stackoverflow.com/questions/57930238/how-to-write-a-function-that-takes-a-folder-name-as-argument-and-return-a-list-o

def lista_arq(caminho):
    files = listdir(caminho)
    return files

caminho = '..'
arqs = lista_arq(caminho)
print(arqs)