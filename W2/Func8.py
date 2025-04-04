#8. Criar uma função que lê de um csv de pares onde o primeiro número é o termo da sequência
#   e o segundo é um número aleatório e plota esses pontos no espaço em forma de um grafo linha.
#   Cada par está em uma linha, e cada elemento do par em uma coluna diferente. Dica: use a biblioteca matplotlib.

import matplotlib.pyplot as plt
import pandas as pd

def plota_pontos():
    df = pd.read_csv("Tests/pares_seq.csv", sep=";", header=None)

    plt.plot(df[0],df[1], marker='o', linestyle='-', color='blue')
    plt.grid(True)
    plt.show()

plota_pontos()