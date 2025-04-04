#2. Criar função para salvar em um arquivo csv pares de números aleatórios. 
#   A função deve receber o valor n (número de pares). A saída é o arquivo csv com os n pares de números aleatórios de zero a um.
#   Cada elemento do par fica em uma coluna. Cada par deve estar em uma linha.

import pandas as pd
import random 

def  salva_par_csv(n):
    dados = []
    for i in range(n):
        dados.append([random.randint(0,1), random.randint(0,1)])
    
    df = pd.DataFrame(dados)
    df.to_csv("Tests/pares.csv", index=False, sep=";", header=False)

salva_par_csv(5)