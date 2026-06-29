import math

class Similitud:
    """Cálculo propio de la similitud del coseno entre vectores TF-IDF.

    Los vectores son los diccionarios dispersos que produce VectorTF_IDF, con
    la forma palabra a peso. La similitud del coseno mide el ángulo entre dos
    vectores: vale 1 cuando apuntan en la misma dirección (documentos muy
    parecidos) y 0 cuando no comparten ninguna palabra.

    El laboratorio exige implementar este cálculo sin usar librerías que ya lo
    provean, por eso se hace a mano sobre los diccionarios.
    """

    def producto_punto(self, vector_a, vector_b):
        """Calcula el producto punto entre dos vectores dispersos.

        Solo aportan las palabras presentes en ambos vectores, así que se
        recorre el más pequeño y se busca cada palabra en el otro.

        Parámetros
        vector_a : dict
            Primer vector con la forma palabra a peso.
        vector_b : dict
            Segundo vector con la forma palabra a peso.

        Retorna
        float
            La suma de los productos de los pesos de las palabras compartidas.
        """
        if len(vector_a) > len(vector_b):
            vector_a, vector_b = vector_b, vector_a
        return sum(peso * vector_b.get(palabra, 0.0)
                   for palabra, peso in vector_a.items())

    def norma(self, vector):
        """Calcula la norma euclidiana (largo) de un vector disperso.

        Parámetros
        vector : dict
            Vector con la forma palabra a peso.

        Retorna
        float
            La raíz de la suma de los pesos al cuadrado.
        """
        return math.sqrt(sum(peso * peso for peso in vector.values()))

    def similitud_coseno(self, vector_a, vector_b):
        """Calcula la similitud del coseno entre dos vectores dispersos.

        Es el producto punto dividido por el producto de las normas. Si alguno
        de los vectores está vacío o tiene norma cero, la similitud es 0.

        Parámetros
        vector_a : dict
            Primer vector con la forma palabra a peso.
        vector_b : dict
            Segundo vector con la forma palabra a peso.

        Retorna
        float
            Un valor entre 0 y 1, donde 1 es máxima similitud.
        """
        if not vector_a or not vector_b:
            return 0.0
        norma_a = self.norma(vector_a)
        norma_b = self.norma(vector_b)
        if norma_a == 0 or norma_b == 0:
            return 0.0
        return self.producto_punto(vector_a, vector_b) / (norma_a * norma_b)
