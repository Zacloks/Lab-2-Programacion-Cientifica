import math

class VectorTF_IDF:
    """Implementación propia de la representación TF-IDF.

    Calcula la importancia de cada palabra en un versículo combinando su
    frecuencia dentro del documento (TF) con cuán rara es en todo el corpus
    (IDF). El laboratorio exige implementar TF-IDF sin librerías que ya lo
    provean, por lo que el cálculo se hace de forma manual.

    El uso típico es llamar a ajustar con todos los documentos para aprender el
    vocabulario y los IDF, y luego a transformar para vectorizar.
    """

    def __init__(self):
        """Inicializa el vocabulario, los IDF y el conteo de documentos."""
        self.vocabulario = {}
        self.idf = {}
        self.totalDocumentos = 0

    def ajustar(self, serie_tokens):
        """Aprende el vocabulario y los valores IDF a partir del corpus.

        Recorre todos los documentos contando en cuántos aparece cada palabra,
        es decir su frecuencia de documento, y calcula el IDF de cada una como
        el logaritmo del total de documentos dividido por esa frecuencia.

        Parámetros
        serie_tokens : Iterable[list[str]]
            Colección de documentos, cada uno una lista de tokens.
        """
        self.totalDocumentos = len(serie_tokens)
        df_counts = {}

        indice = 0
        for tokens in serie_tokens:
            tokensUnicos = set(tokens)
            for token in tokensUnicos:
                if token not in self.vocabulario:
                    self.vocabulario[token] = indice
                    indice += 1
                df_counts[token] = df_counts.get(token, 0) + 1

        for token, df in df_counts.items():
            self.idf[token] = math.log(self.totalDocumentos / df)

    def calcularTF_IDF_Documento(self, tokens):
        """Calcula el vector TF-IDF de un único documento.

        El TF de cada término es su frecuencia relativa dentro del documento,
        o sea sus apariciones divididas por el total de términos, multiplicada
        por el IDF aprendido.

        Parámetros
        tokens : list[str]
            Tokens de un documento.

        Retorna
        dict
            Mapeo de cada palabra a su peso TF-IDF, en representación dispersa.
            Un documento vacío devuelve un diccionario vacío.
        """
        tf_Documento = {}
        totalTerminos = len(tokens)

        if totalTerminos == 0:
            return {}

        for token in tokens:
            tf_Documento[token] = tf_Documento.get(token, 0) + 1

        tfIdf_Documento = {}
        for token, frecuencia in tf_Documento.items():
            if token in self.vocabulario:
                tf = frecuencia / totalTerminos
                tfIdf_Documento[token] = tf * self.idf[token]

        return tfIdf_Documento

    def transformar(self, serie_tokens):
        """Vectoriza una serie de documentos con el modelo ya ajustado.

        Parámetros
        serie_tokens : pandas.Series
            Serie cuyos elementos son listas de tokens.

        Retorna
        pandas.Series
            Serie de diccionarios TF-IDF, uno por documento.
        """
        return serie_tokens.apply(self.calcularTF_IDF_Documento)
