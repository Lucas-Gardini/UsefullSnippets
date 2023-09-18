# C++

## Número primo

```cpp
#include <iostream>

bool is_prime(int num) {
    if (num <= 1) {
        return false;
    } else if (num <= 3) {
        return true;
    } else if (num % 2 == 0 || num % 3 == 0) {
        return false;
    }
    int i = 5;
    while (i * i <= num) {
        if (num % i == 0 || num % (i + 2) == 0) {
            return false;
        }
        i += 6;
    }
    return true;
}

```

## MergeSort [E]

```cpp
#include <iostream>
#include <vector>

// Não invocar diretamente
std::vector<int> merge(std::vector<int>& left, std::vector<int>& right) {
    std::vector<int> result;
    size_t i = 0, j = 0;

    while (i < left.size() && j < right.size()) {
        if (left[i] <= right[j]) {
            result.push_back(left[i]);
            i++;
        } else {
            result.push_back(right[j]);
            j++;
        }
    }

    while (i < left.size()) {
        result.push_back(left[i]);
        i++;
    }

    while (j < right.size()) {
        result.push_back(right[j]);
        j++;
    }

    return result;
}

std::vector<int> merge_sort(std::vector<int>& arr) {
    if (arr.size() <= 1) {
        return arr;
    }

    size_t mid = arr.size() / 2;
    std::vector<int> left(arr.begin(), arr.begin() + mid);
    std::vector<int> right(arr.begin() + mid, arr.end());

    left = merge_sort(left);
    right = merge_sort(right);

    return merge(left, right);
}

```

## QuickSort [NE]

```cpp
#include <iostream>
#include <vector>

std::vector<int> quicksort(std::vector<int>& arr) {
    if (arr.size() <= 1) {
        return arr;
    }

    int pivot = arr[arr.size() / 2];
    std::vector<int> left, middle, right;

    for (int num : arr) {
        if (num < pivot) {
            left.push_back(num);
        } else if (num == pivot) {
            middle.push_back(num);
        } else {
            right.push_back(num);
        }
    }

    std::vector<int> sorted_left = quicksort(left);
    std::vector<int> sorted_right = quicksort(right);

    sorted_left.insert(sorted_left.end(), middle.begin(), middle.end());
    sorted_left.insert(sorted_left.end(), sorted_right.begin(), sorted_right.end());

    return sorted_left;
}

```

## Progressão aritmética

```cpp
#include <iostream>
#include <vector>

std::vector<int> progressao_aritmetica(int a1, int r, int n) {
    std::vector<int> termos;

    for (int i = 0; i < n; i++) {
        int proximo_termo = a1 + i * r;
        termos.push_back(proximo_termo);
    }

    return termos;
}

int main() {
    int a1 = 2;  // Primeiro termo
    int r = 3;   // Razão da progressão
    int n = 5;   // Número de termos

    std::vector<int> pa = progressao_aritmetica(a1, r, n);

    std::cout << "Progressão Aritmética:";
    for (int termo : pa) {
        std::cout << " " << termo;
    }
    std::cout << std::endl;

    return 0;
}


```

## Progressão geométrica

```cpp
#include <iostream>
#include <vector>

std::vector<int> progressao_geometrica(int a1, int razao, int n) {
    std::vector<int> termos;
    termos.push_back(a1);
    for (int i = 1; i < n; i++) {
        int proximo_termo = termos.back() * razao;
        termos.push_back(proximo_termo);
    }
    return termos;
}

int main() {
    int a1 = 2;     // Primeiro termo
    int razao = 3;  // Razão da progressão
    int n = 5;      // Número de termos

    std::vector<int> pg = progressao_geometrica(a1, razao, n);

    std::cout << "Progressão Geométrica:";
    for (int termo : pg) {
        std::cout << " " << termo;
    }
    std::cout << std::endl;

    return 0;
}


```

## Árvore binária

```cpp
#include <iostream>

class No {
public:
    int valor;
    No* filho_esquerdo;
    No* filho_direito;

    No(int v) : valor(v), filho_esquerdo(nullptr), filho_direito(nullptr) {}
};

int main() {
    No* raiz = new No(1);
    raiz->filho_esquerdo = new No(2);
    raiz->filho_direito = new No(3);

    // Fazer operações com a árvore aqui

    delete raiz->filho_esquerdo;
    delete raiz->filho_direito;
    delete raiz;

    return 0;
}


```

## Árvore binária de busca (BST)

```cpp
#include <iostream>

class NoBST {
public:
    int valor;
    NoBST* filho_esquerdo;
    NoBST* filho_direito;

    NoBST(int v) : valor(v), filho_esquerdo(nullptr), filho_direito(nullptr) {}
};

class ArvoreBinariaDeBusca {
public:
    NoBST* raiz;

    ArvoreBinariaDeBusca() : raiz(nullptr) {}

    void inserir(int valor) {
        raiz = inserir_recursivamente(raiz, valor);
    }

    NoBST* inserir_recursivamente(NoBST* no, int valor) {
        if (no == nullptr) {
            return new NoBST(valor);
        }

        if (valor < no->valor) {
            no->filho_esquerdo = inserir_recursivamente(no->filho_esquerdo, valor);
        } else {
            no->filho_direito = inserir_recursivamente(no->filho_direito, valor);
        }

        return no;
    }

    NoBST* buscar(int valor) {
        return buscar_recursivamente(raiz, valor);
    }

    NoBST* buscar_recursivamente(NoBST* no, int valor) {
        if (no == nullptr || no->valor == valor) {
            return no;
        }

        if (valor < no->valor) {
            return buscar_recursivamente(no->filho_esquerdo, valor);
        } else {
            return buscar_recursivamente(no->filho_direito, valor);
        }
    }
};

int main() {
    ArvoreBinariaDeBusca arvore;
    arvore.inserir(10);
    arvore.inserir(5);
    arvore.inserir(15);

    NoBST* resultado = arvore.buscar(5);
    if (resultado) {
        std::cout << "Valor 5 encontrado na árvore!" << std::endl;
    } else {
        std::cout << "Valor 5 não encontrado na árvore." << std::endl;
    }

    return 0;
}

```

## Grafo

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <limits>

class Grafo {
public:
    std::map<std::string, std::map<std::string, int>> vertices;

    void adicionar_vertice(const std::string& vertice) {
        vertices[vertice] = std::map<std::string, int>();
    }

    void adicionar_aresta(const std::string& origem, const std::string& destino, int peso) {
        vertices[origem][destino] = peso;
        vertices[destino][origem] = peso;
    }

    std::string menor_caminho(const std::string& origem, const std::string& destino) {
        std::set<std::string> visitados;
        std::map<std::string, int> distancias;
        std::map<std::string, std::string> caminho_anterior;

        for (const auto& pair : vertices) {
            distancias[pair.first] = std::numeric_limits<int>::max();
        }

        distancias[origem] = 0;

        while (visitados.size() < vertices.size()) {
            std::string vertice_atual = "";
            for (const auto& pair : vertices) {
                if (visitados.find(pair.first) == visitados.end()) {
                    if (vertice_atual.empty() || distancias[pair.first] < distancias[vertice_atual]) {
                        vertice_atual = pair.first;
                    }
                }
            }

            visitados.insert(vertice_atual);

            for (const auto& pair : vertices[vertice_atual]) {
                std::string vizinho = pair.first;
                int peso = pair.second;

                int distancia = distancias[vertice_atual] + peso;

                if (distancia < distancias[vizinho]) {
                    distancias[vizinho] = distancia;
                    caminho_anterior[vizinho] = vertice_atual;
                }
            }
        }

        std::vector<std::string> caminho;
        std::string atual = destino;

        while (atual != origem) {
            caminho.push_back(atual);
            atual = caminho_anterior[atual];
        }

        caminho.push_back(origem);
        std::reverse(caminho.begin(), caminho.end());

        std::string caminho_com_pesos;
        for (const std::string& vertice : caminho) {
            caminho_com_pesos += vertice + " (" + std::to_string(distancias[vertice]) + "), ";
        }

        // Remover a vírgula e o espaço extra no final
        caminho_com_pesos = caminho_com_pesos.substr(0, caminho_com_pesos.size() - 2);

        return caminho_com_pesos;
    }
};

int main() {
    Grafo grafo;
    grafo.adicionar_vertice("A");
    grafo.adicionar_vertice("B");
    grafo.adicionar_vertice("C");
    grafo.adicionar_aresta("A", "B", 1);
    grafo.adicionar_aresta("B", "C", 2);
    grafo.adicionar_aresta("A", "C", 4);

    std::string caminho = grafo.menor_caminho("A", "C");
    std::cout << "Menor caminho de A para C: " << caminho << std::endl;

    return 0;
}

```
