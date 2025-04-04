#6. Criar uma função que lê de um txt de pares de números aleatórios e plota esses pontos no espaço. 
#   Cada par está em uma linha,e cada elemento do par está separado por um ponto e vírgula (;). Dica: use a biblioteca matplotlib.

import matplotlib.pyplot as plt


def plota_pontos():
    x = []
    y = []

    with open("Tests/pares.txt", "r") as file:
        for linha in file:
            linha = linha.strip()
            valores = linha.split(";")
            x.append(int(valores[0]))
            y.append(int(valores[1]))


    plt.scatter(x, y, color='red', s=10)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.title("Pontos gerados aleatoriamente")
    plt.grid(True)
    plt.show()

plota_pontos()