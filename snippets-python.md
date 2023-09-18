# Python

## Número primo

```py
def is_prime(num):
    if num <= 1:
        return False
    elif num <= 3:
        return True
    elif num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True
```

## MergeSort [E]

```py
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

# Não invocar manualmente
def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

```

## QuickSort [NE]

```py
def quicksort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)
```

## Progressão aritmética

```py
def progressao_aritmetica(a1, r, n):
    termos = []

    # Loop de 0 a n-1 (os primeiros n termos)
    for i in range(n):
        # Calcula o próximo termo e o adiciona à lista
        proximo_termo = a1 + i * r
        termos.append(proximo_termo)

    return termos

```

## Progressão geométrica

```py
def progressao_geometrica(a1, razao, n):
    termos = [a1]
    for _ in range(n - 1):
        proximo_termo = termos[-1] * razao
        termos.append(proximo_termo)
    return termos

# Exemplo de uso:
a1 = 2  # Primeiro termo
razao = 3  # Razão da progressão
n = 5  # Número de termos

pg = progressao_geometrica(a1, razao, n)
print("Progressão Geométrica:", pg)

```

## Árvore binária

```py
class No:
    def __init__(self, valor):
        self.valor = valor
        self.filho_esquerdo = None
        self.filho_direito = None

raiz = No(1)
raiz.filho_esquerdo = No(2)
raiz.filho_direito = No(3)

```

## Árvore binária de busca (BST)

```py
class NoBST:
    def __init__(self, valor):
        self.valor = valor
        self.filho_esquerdo = None
        self.filho_direito = None

class ArvoreBinariaDeBusca:
    def __init__(self):
        self.raiz = None

    def inserir(self, valor):
        self.raiz = self._inserir_recursivamente(self.raiz, valor)

    def _inserir_recursivamente(self, no, valor):
        if no is None:
            return NoBST(valor)

        if valor < no.valor:
            no.filho_esquerdo = self._inserir_recursivamente(no.filho_esquerdo, valor)
        else:
            no.filho_direito = self._inserir_recursivamente(no.filho_direito, valor)

        return no

    def buscar(self, valor):
        return self._buscar_recursivamente(self.raiz, valor)

    def _buscar_recursivamente(self, no, valor):
        if no is None or no.valor == valor:
            return no

        if valor < no.valor:
            return self._buscar_recursivamente(no.filho_esquerdo, valor)
        else:
            return self._buscar_recursivamente(no.filho_direito, valor)

arvore = ArvoreBinariaDeBusca()
arvore.inserir(10)
arvore.inserir(5)
arvore.inserir(15)

resultado = arvore.buscar(5)
if resultado:
    print(f"Valor 5 encontrado na árvore!")
else:
    print("Valor 5 não encontrado na árvore.")

```

## Grafo

```py
class Grafo:
    def __init__(self):
        self.vertices = {}

    def adicionar_vertice(self, vertice):
        self.vertices[vertice] = {}

    def adicionar_aresta(self, origem, destino, peso):
        self.vertices[origem][destino] = peso
        self.vertices[destino][origem] = peso

    def menor_caminho(self, origem, destino):
        visitados = set()
        distancias = {vertice: float('inf') for vertice in self.vertices}
        distancias[origem] = 0
        caminho_anterior = {}

        while len(visitados) < len(self.vertices):
            vertice_atual = None
            for vertice in self.vertices:
                if vertice not in visitados:
                    if vertice_atual is None:
                        vertice_atual = vertice
                    elif distancias[vertice] < distancias[vertice_atual]:
                        vertice_atual = vertice

            for vizinho, peso in self.vertices[vertice_atual].items():
                distancia = distancias[vertice_atual] + peso
                if distancia < distancias[vizinho]:
                    distancias[vizinho] = distancia
                    caminho_anterior[vizinho] = vertice_atual

            visitados.add(vertice_atual)

        caminho = [destino]
        while destino != origem:
            destino = caminho_anterior[destino]
            caminho.append(destino)

        caminho.reverse()
        caminho_com_pesos = [c + f" ({distancias[c]})" for c in caminho]
        return ", ".join(caminho_com_pesos)

# Exemplo de uso:
grafo = Grafo()
grafo.adicionar_vertice("A")
grafo.adicionar_vertice("B")
grafo.adicionar_vertice("C")
grafo.adicionar_aresta("A", "B", 1)
grafo.adicionar_aresta("B", "C", 2)
grafo.adicionar_aresta("A", "C", 4)

caminho = grafo.menor_caminho("A", "C")
print("Menor caminho de A para C:", caminho)

```
