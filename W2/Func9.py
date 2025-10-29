#9. Listar todos os documentos de uma pasta e suas subpastas

import os
def lista_pastas(caminho):
    lista = []
    for raiz, pastas, _ in os.walk(caminho):
        for pasta in pastas:
            lista.append(os.path.join(raiz, pasta))
    return lista

def lista_txts(caminho):
    arquivos_txt = []
    for raiz, _, arquivos in os.walk(caminho):
        for arquivo in arquivos:
            if arquivo.endswith(".txt"):
                caminho_completo = os.path.join(raiz, arquivo)
                arquivos_txt.append(caminho_completo)
    return arquivos_txt

caminho = '../IC'
pastas = lista_pastas(caminho)
print(pastas)

arqs_txt = lista_txts(caminho)
for arquivo in arqs_txt:
    print(arquivo )