#4. Criar função para salvar em um arquivo csv pares com um número não aleatório e um número aleatório. 
#   A função deve receber o valor n (número de pares). A saída é o arquivo csv com os n pares, 
#   onde o primeiro é referente ao termo da sequência e o segundo um número aleatório de zero a um.
#   Cada elemento do par fica em uma coluna. Cada par deve estar em uma linha. O primeiro termo vai de 1 até n.
import random
import pandas as pd

def  salva_par_csv(n):
    dados = []
    for i in range(n):
        dados.append([i+1, random.uniform(0.0,1.0)])
    
    df = pd.DataFrame(dados)
    df.to_csv("Tests/pares_seq.csv", index=False, sep=";", header=False)

salva_par_csv(35)