#7. Criar uma função que lê de um txt de pares onde o primeiro número é o termo da sequência 
#   e o segundo é um número aleatório e plota esses pontos no espaço em forma de um grafo linha.
#   Cada par está em uma linha, e cada elemento do par é separado por ponto e vírgula (;). Dica: use a biblioteca matplotlib.

import matplotlib.pyplot as plt

def plota_grafico():
    x = []
    y = []

    with open("Tests/pares_seq.txt", "r") as file:
        for linha in file:
            linha = linha.strip()
            valores = linha.split(";")
            x.append(int(valores[0]))
            y.append(int(valores[1]))


    plt.plot(x, y, marker='o', linestyle='-', color='blue')
    plt.grid(True)
    plt.show()

plota_grafico()