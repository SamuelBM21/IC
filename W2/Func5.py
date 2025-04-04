#5. Criar uma função que lê de um csv de pares de números aleatórios e plota esses pontos no espaço. 
#   Cada par está em uma linha, e cada elemento do par em uma coluna diferente. Dica: use a biblioteca matplotlib.

import matplotlib.pyplot as plt
import pandas as pd

def plota_pontos():
    df = pd.read_csv("Tests/pares.csv", sep=";", header=None)

    plt.scatter(df[0], df[1], color='blue', s=100)
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.title("Pontos gerados aleatoriamente")
    plt.grid(True)
    plt.show()

plota_pontos()