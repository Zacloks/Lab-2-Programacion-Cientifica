import math

class VectorTF_IDF:
    """TF-IDF propio para representar versículos como vectores.

    Combina la frecuencia de cada palabra en el documento (TF) con lo rara que
    es en el corpus (IDF). La pauta pide implementarlo sin librerías. Se usa
    llamando a ajustar y después a transformar.
    """

    def __init__(self):
        """Deja vacíos el vocabulario, los IDF y el conteo de documentos."""
        self.vocabulario = {}
        self.idf = {}
        self.totalDocumentos = 0

    def ajustar(self, serie_tokens):
        """Aprende el vocabulario y los IDF a partir del corpus.

        Cuenta en cuántos documentos aparece cada palabra y calcula su IDF como
        el log del total de documentos dividido por esa cuenta.

        Parámetros
        serie_tokens : Iterable[list[str]]
            Documentos, cada uno una lista de tokens.
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
        """Calcula el vector TF-IDF de un documento.

        El TF es la frecuencia relativa de cada palabra en el documento,
        multiplicada por su IDF.

        Parámetros
        tokens : list[str]
            Tokens del documento.

        Retorna
        dict
            Palabra a peso TF-IDF. Un documento vacío devuelve un diccionario
            vacío.
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
            Serie de listas de tokens.

        Retorna
        pandas.Series
            Serie de diccionarios TF-IDF, uno por documento.
        """
        return serie_tokens.apply(self.calcularTF_IDF_Documento)
