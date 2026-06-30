import math

class SimilitudCoseno:
    """Similitud del coseno propia entre vectores TF-IDF.

    La pauta pide implementarla sin librerías. Trabaja sobre los diccionarios
    dispersos del TF-IDF. Vale 1 cuando dos vectores apuntan igual y 0 cuando no
    comparten palabras. La usan el buscador (similitud_coseno) y el heatmap
    (matriz_similitud).
    """

    def producto_punto(self, vector_a, vector_b):
        """Calcula el producto punto entre dos vectores dispersos.

        Recorre el vector más corto y solo suma las palabras que están en ambos.

        Parámetros
        vector_a, vector_b : dict
            Vectores TF-IDF (palabra a peso).

        Retorna
        float
            Suma de los productos de los pesos compartidos.
        """
        if len(vector_a) > len(vector_b):
            vector_a, vector_b = vector_b, vector_a
        return sum(peso * vector_b.get(termino, 0.0)
                   for termino, peso in vector_a.items())

    def norma(self, vector):
        """Calcula la norma euclidiana de un vector disperso.

        Parámetros
        vector : dict
            Vector TF-IDF.

        Retorna
        float
            Raíz de la suma de los pesos al cuadrado.
        """
        return math.sqrt(sum(peso * peso for peso in vector.values()))

    def similitud_coseno(self, vector_a, vector_b):
        """Calcula la similitud del coseno entre dos vectores.

        Si alguno es vacío o de norma cero devuelve 0, para no dividir por cero.

        Parámetros
        vector_a, vector_b : dict
            Vectores a comparar.

        Retorna
        float
            Similitud en el rango [0, 1].
        """
        if not vector_a or not vector_b:
            return 0.0
        norma_a = self.norma(vector_a)
        norma_b = self.norma(vector_b)
        if norma_a == 0 or norma_b == 0:
            return 0.0
        return self.producto_punto(vector_a, vector_b) / (norma_a * norma_b)

    def matriz_similitud(self, vectores):
        """Construye la matriz NxN de similitud entre N vectores.

        Es simétrica y con diagonal 1. Calcula solo el triángulo superior y lo
        refleja.

        Parámetros
        vectores : list[dict]
            Lista de N vectores TF-IDF.

        Retorna
        list[list[float]]
            Matriz NxN de similitudes.
        """
        n = len(vectores)
        normas = [self.norma(vector) for vector in vectores]
        matriz = [[0.0] * n for _ in range(n)]

        for i in range(n):
            matriz[i][i] = 1.0 if normas[i] > 0 else 0.0
            for j in range(i + 1, n):
                if normas[i] == 0 or normas[j] == 0:
                    similitud = 0.0
                else:
                    similitud = (self.producto_punto(vectores[i], vectores[j]) / (normas[i] * normas[j]))
                matriz[i][j] = similitud
                matriz[j][i] = similitud

        return matriz
