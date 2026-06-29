import math

class SimilitudCoseno:
    """Implementación propia de la similitud del coseno entre vectores TF-IDF.

    El laboratorio exige calcular la similitud del coseno sin usar librerías que
    ya la provean, por lo que aquí se implementa de forma manual. Trabaja sobre
    los vectores dispersos que produce VectorTF_IDF, es decir diccionarios de la
    forma palabra a peso TF-IDF, en los que las palabras ausentes valen cero.

    La similitud del coseno entre dos vectores es el coseno del ángulo que
    forman: vale 1 cuando apuntan en la misma dirección (documentos idénticos en
    contenido), 0 cuando no comparten ningún término y nunca es negativa porque
    los pesos TF-IDF no lo son. Se calcula como el producto punto dividido por el
    producto de las normas.
    """

    def producto_punto(self, vector_a, vector_b):
        """Calcula el producto punto entre dos vectores dispersos.

        Solo contribuyen los términos presentes en ambos vectores, así que se
        recorre el más pequeño y se buscan sus términos en el más grande.

        Parámetros
        vector_a, vector_b : dict
            Vectores TF-IDF dispersos (palabra a peso).

        Retorna
        float
            Suma de los productos de los pesos de los términos compartidos.
        """
        if len(vector_a) > len(vector_b):
            vector_a, vector_b = vector_b, vector_a
        return sum(peso * vector_b.get(termino, 0.0)
                   for termino, peso in vector_a.items())

    def norma(self, vector):
        """Calcula la norma euclidiana (longitud) de un vector disperso.

        Parámetros
        vector : dict
            Vector TF-IDF disperso.

        Retorna
        float
            Raíz de la suma de los cuadrados de los pesos.
        """
        return math.sqrt(sum(peso * peso for peso in vector.values()))

    def similitud(self, vector_a, vector_b):
        """Calcula la similitud del coseno entre dos vectores dispersos.

        Si alguno de los vectores es nulo (norma cero, por ejemplo un versículo
        que quedó sin tokens tras el preprocesamiento) la similitud se define
        como cero para evitar una división por cero.

        Parámetros
        vector_a, vector_b : dict
            Vectores TF-IDF dispersos a comparar.

        Retorna
        float
            Similitud del coseno en el rango [0, 1].
        """
        norma_a = self.norma(vector_a)
        norma_b = self.norma(vector_b)
        if norma_a == 0 or norma_b == 0:
            return 0.0
        return self.producto_punto(vector_a, vector_b) / (norma_a * norma_b)

    def ranking(self, vector_consulta, vectores):
        """Ordena un conjunto de vectores por su similitud a una consulta.

        Útil para un motor de búsqueda: dada la representación de una frase,
        devuelve los índices de los documentos del más al menos parecido.

        Parámetros
        vector_consulta : dict
            Vector TF-IDF de la consulta.
        vectores : Iterable[dict]
            Vectores TF-IDF de los documentos del corpus.

        Retorna
        list[tuple[int, float]]
            Pares (índice, similitud) ordenados de mayor a menor similitud.
        """
        puntajes = [(indice, self.similitud(vector_consulta, vector))
                    for indice, vector in enumerate(vectores)]
        puntajes.sort(key = lambda par: par[1], reverse = True)
        return puntajes

    def matriz_similitud(self, vectores):
        """Construye la matriz NxN de similitud del coseno entre N vectores.

        La matriz es simétrica y su diagonal vale 1 (cada documento es idéntico
        a sí mismo), salvo para documentos nulos cuya diagonal es 0. Se calcula
        solo el triángulo superior y se refleja, evitando trabajo redundante.

        Parámetros
        vectores : list[dict]
            Lista de N vectores TF-IDF dispersos.

        Retorna
        list[list[float]]
            Matriz NxN de similitudes del coseno.
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
                    similitud = (self.producto_punto(vectores[i], vectores[j])
                                 / (normas[i] * normas[j]))
                matriz[i][j] = similitud
                matriz[j][i] = similitud

        return matriz