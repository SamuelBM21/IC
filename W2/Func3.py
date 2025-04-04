#3. Criar função para salvar em um arquivo txt pares com um número não aleatório e um número aleatório. 
#   A função deve receber o valor n (número de pares). A saída é o arquivo txt com os n pares, 
#   onde o primeiro é referente ao termo da sequência e o segundo um número aleatório de zero a um.
#   O padrão de escrita para o par é número1;número2. Cada par deve estar em uma linha. O primeiro termo vai de 1 até n.

import random

def salva_par(n):
    with open("Tests/pares_seq.txt","w") as file:
        for i in range(n):
            numero1 = i + 1
            numero2 = random.randint(0, 1)
            file.write(str(numero1) + ";" + str(numero2) + "\n")


salva_par(35)