#   1.Criar função para salvar em um arquivo txt pares de números aleatórios. A função deve receber o valor n (número de pares).
#   A saída é o arquivo txt com os n pares de números aleatórios de zero a um. 
#   O padrão de escrita para o par é número1;número2. Cada par deve estar em uma linha.

import random

def salva_par(n):
    with open("Tests/pares.txt","w") as file:
        for i in range(n):
            numero1 = random.uniform(0.0, 1.0)
            numero2 = random.uniform(0.0, 1.0)
            file.write(str(numero1) + ";" + str(numero2) + "\n")


salva_par(5)