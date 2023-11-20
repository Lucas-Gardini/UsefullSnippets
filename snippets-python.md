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

## Grafo (Busca em Largura - BFS)

O BFS é usado para percorrer grafos em largura.

```py
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited

# Exemplo de uso:
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F'},
    'D': {'B'},
    'E': {'B', 'F'},
    'F': {'C', 'E'}
}
start_node = 'A'
visited = bfs(graph, start_node)
print("Nós visitados em ordem de BFS:", visited)
```

## Operações com dicionário

### Ordenar pelo valor

```py
x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
x_sorted = dict(sorted(x.items(), key=lambda item: item[1]))
# result: {0: 0, 2: 1, 1: 2, 4: 3, 3: 4}
```

### Ordenar pela chave

```py
import collections

x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}

sorted_x = sorted(x.items(), key=lambda kv: kv[1]) # Saída é uma tupla

sorted_dict = collections.OrderedDict(sorted_x) # Retorna um dicionário ordenado
```

## Busca binária

A busca binária é um algoritmo eficiente para encontrar um elemento em um array ordenado.

```py
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Exemplo de uso:
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target = 6
result = binary_search(arr, target)
if result != -1:
    print(f"O elemento {target} está na posição {result}.")
else:
    print("O elemento não foi encontrado.")

```

## Programação dinâmica

A programação dinâmica é uma técnica usada para resolver problemas de otimização.

```
def fibonacci(n):
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]

# Exemplo de uso:
n = 10
result = fibonacci(n)
print(f"O {n}-ésimo número de Fibonacci é {result}.")
```

## Árvore de segmentos

A árvore de segmentos é usada para consultas de intervalo em arrays.

```py
class SegmentTree:
    def __init__(self, arr):
        self.arr = arr
        self.tree = [0] * (4 * len(arr))

    def build(self, node, start, end):
        if start == end:
            self.tree[node] = self.arr[start]
            return
        mid = (start + end) // 2
        self.build(2 * node, start, mid)
        self.build(2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, node, start, end, left, right):
        if left > end or right < start:
            return 0
        if left <= start and right >= end:
            return self.tree[node]
        mid = (start + end) // 2
        left_sum = self.query(2 * node, start, mid, left, right)
        right_sum = self.query(2 * node + 1, mid + 1, end, left, right)
        return left_sum + right_sum

# Exemplo de uso:
arr = [1, 3, 5, 7, 9, 11, 13]
segment_tree = SegmentTree(arr)
segment_tree.build(1, 0, len(arr) - 1)
left_index, right_index = 1, 4
result = segment_tree.query(1, 0, len(arr) - 1, left_index, right_index)
print(f"A soma no intervalo [{left_index}, {right_index}] é {result}.")
```

## Algoritmo de Dijkstra

O algoritmo de Dijkstra é usado para encontrar o caminho mais curto em um grafo ponderado.

```py
import heapq

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    queue = [(0, start)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    return distances

# Exemplo de uso:
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
start_node = 'A'
shortest_distances = dijkstra(graph, start_node)
print("Distâncias mais curtas a partir de A:", shortest_distances)
```

## Algoritmo de Kruskal

O algoritmo de Kruskal é usado para encontrar a árvore geradora mínima de um grafo ponderado.

```py
def kruskal(graph):
    def find(parent, node):
        if parent[node] == node:
            return node
        return find(parent, parent[node])

    def union(parent, x, y):
        x_root = find(parent, x)
        y_root = find(parent, y)
        parent[x_root] = y_root

    edges = []
    for node in graph:
        for neighbor, weight in graph[node].items():
            edges.append((weight, node, neighbor))
    edges.sort()

    minimum_spanning_tree = {}
    parent = {node: node for node in graph}

    for weight, u, v in edges:
        if find(parent, u) != find(parent, v):
            union(parent, u, v)
            if u in minimum_spanning_tree:
                minimum_spanning_tree[u][v] = weight
            else:
                minimum_spanning_tree[u] = {v: weight}
    return minimum_spanning_tree


# Exemplo de uso:
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
minimum_spanning_tree = kruskal(graph)
print("Árvore Geradora Mínima:", minimum_spanning_tree)
```

## Algoritmo de Kadane para Maior Subarray Contíguo (Subarray Máximo)

Este algoritmo encontra o maior subarray contíguo em uma lista de números.

```py
def max_subarray(arr):
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# Exemplo de uso:
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
max_sum = max_subarray(arr)
print("Maior subarray contíguo:", max_sum)
```

## Algoritmo de Ordenação por Contagem (Counting Sort)

O Counting Sort é um algoritmo de ordenação eficiente para números inteiros em um intervalo específico.

```py
def counting_sort(arr):
    max_val = max(arr)
    min_val = min(arr)
    count = [0] * (max_val - min_val + 1)
    output = [0] * len(arr)
    for num in arr:
        count[num - min_val] += 1
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    for num in arr:
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1
    return output

# Exemplo de uso:
arr = [4, 2, 2, 8, 3, 3, 1]
sorted_arr = counting_sort(arr)
print("Lista ordenada:", sorted_arr)
```

## Algoritmo de Busca de Substring (KMP)

O algoritmo KMP é usado para encontrar todas as ocorrências de uma substring em uma string.

```py
def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        j = 0
        for i in range(1, len(pattern)):
            while j > 0 and pattern[i] != pattern[j]:
                j = lps[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
            lps[i] = j
        return lps

    lps = build_lps(pattern)
    i = j = 0
    matches = []
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return matches

# Exemplo de uso:
text = "ABABDABACDABABCABAB"
pattern = "ABABCABAB"
positions = kmp_search(text, pattern)
print("Posições das ocorrências:", positions)
```

## Menor Quantidade de Cédulas

Código para calcular a menor quantidade de cédulas para um determinado valor.

```py
def menor_quantidade_cedulas(N, valores_cedulas, valor_cheque):
    # Aplicando Counting Sort para ordenar as cédulas
    counting_sort = [0] * (max(valores_cedulas) + 1)
    for valor in valores_cedulas:
        counting_sort[valor] += 1

    # Calculando a quantidade mínima de cédulas
    quantidade_cedulas = 0
    for valor in reversed(range(len(counting_sort))):
        qtd_notas = min(valor_cheque // valor, counting_sort[valor])
        quantidade_cedulas += qtd_notas
        valor_cheque -= qtd_notas * valor

    return quantidade_cedulas

# Exemplo de uso
N = 3
valores_cedulas = [5, 1, 10]
valor_cheque = 28
resultado = menor_quantidade_cedulas(N, valores_cedulas, valor_cheque)
print(resultado)
```

## Árvore Genealógica e Verificação de Parentesco

Código para verificar se duas pessoas são parentes.

```py
class Pessoa:
    def __init__(self, nome):
        self.nome = nome
        self.pais = set()

def construir_arvore_pessoas(relacoes):
    pessoas = {}

    for pai1, pai2, filho in relacoes:
        if pai1 not in pessoas:
            pessoas[pai1] = Pessoa(pai1)
        if pai2 not in pessoas:
            pessoas[pai2] = Pessoa(pai2)
        if filho not in pessoas:
            pessoas[filho] = Pessoa(filho)

        pessoas[filho].pais.add(pai1)
        pessoas[filho].pais.add(pai2)

    return pessoas

def tem_parentesco(arvore_pessoas, pessoa1, pessoa2):
    visitados = set()

    def dfs(pessoa, alvo):
        if pessoa == alvo:
            return True
        visitados.add(pessoa)
        for pai in arvore_pessoas[pessoa].pais:
            if pai not in visitados and dfs(pai, alvo):
                return True
        return False

    return dfs(pessoa1, pessoa2) or dfs(pessoa2, pessoa1)

# Entrada
N, C, T = map(int, input().split())

relacoes_parentesco = [input().split() for _ in range(C)]

arvore_pessoas = construir_arvore_pessoas(relacoes_parentesco)

# Processamento dos casos de teste
for _ in range(T):
    pessoa1, pessoa2 = input().split()
    resultado = tem_parentesco(arvore_pessoas, pessoa1, pessoa2)
    print("verdadeiro" if resultado else "falso")
```

Exemplos de Entradas
11 6 5
Ana Ivo Eva
Bia Gil Rai
Bia Gil Clo
Bia Gil Ary
Eva Rai Noe
Ary Lia Gal
Eva Ary
Noe Gal
Lia Rai
Lia Noe
Gal Rai

Exemplos de Saídas
falso
verdadeiro
falso
falso
verdadeir

## Função de Verificação de Palíndromo

```py
def pode_formar_palindromo(s):
    # Cria um dicionário para contar a frequência de cada caractere na string
    contagem_caracteres = {}
    for char in s:
        contagem_caracteres[char] = contagem_caracteres.get(char, 0) + 1

    # Conta quantos caracteres têm uma contagem ímpar
    contagem_impares = sum(1 for contagem in contagem_caracteres.values() if contagem % 2 != 0)

    # Se tiver no máximo um caractere com contagem ímpar, pode formar um palíndromo
    return contagem_impares <= 1
```

## Função para Contar Palavras

```py
def count_words(grid, words):
    word_counts = {word: 0 for word in words}

    rows = len(grid)
    cols = len(grid[0])

    # Horizontal
    for row in range(rows):
        for col in range(cols):
            for word in words:
                # Verifica se a palavra aparece na horizontal da esquerda para a direita
                if ''.join(grid[row][col:col + len(word)]) == word:
                    word_counts[word] += 1

    # Vertical
    for col in range(cols):
        for row in range(rows):
            for word in words:
                # Cria uma string vertical para verificar a palavra de cima para baixo
                vertical_word = ''.join(grid[row + k][col] for k in range(len(word))) if row + len(word) <= rows else ''
                if vertical_word == word:
                    word_counts[word] += 1

    return word_counts

# Exemplo de entrada
matrix_dim = input().split()
rows = int(matrix_dim[0])
cols = int(matrix_dim[1])

matrix = []
for _ in range(rows):
    row = input().split()
    matrix.append(row)

words_to_find = input().split()

# Contagem das palavras na grade de letras
word_counts = count_words(matrix, words_to_find)

# Exibição dos resultados
for word in words_to_find:
    count = word_counts[word]
    print(f"{word}: {count}")

```
