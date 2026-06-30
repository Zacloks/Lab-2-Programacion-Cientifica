from src.preprocesamiento.preprocesador_texto import PreprocesadorTexto

class BuscadorSemantico:
    """Motor de búsqueda semántico sobre el corpus.

    Dada una frase del usuario o un versículo, devuelve los K versículos más
    parecidos según la similitud del coseno sobre vectores TF-IDF. Al
    construirse precomputa los vectores de todos los versículos para que cada
    búsqueda sea rápida.
    """

    def __init__(self, df, tfidf, similitud, preprocesador = None):
        """Ajusta el TF-IDF y vectoriza el corpus.

        Parámetros
        df : pandas.DataFrame
            Corpus preprocesado, con tokens y los datos de cada versículo.
        tfidf : VectorTF_IDF
            TF-IDF propio. Si no está ajustado, se ajusta aquí.
        similitud : SimilitudCoseno
            Calcula la similitud del coseno.
        preprocesador : PreprocesadorTexto o None
            Limpia la consulta. Si es None se crea uno nuevo para procesarla
            igual que el corpus.
        """
        self.df = df.reset_index(drop = True)
        self.tfidf = tfidf
        self.similitud = similitud
        self.preprocesador = preprocesador if preprocesador is not None else PreprocesadorTexto()

        if not self.tfidf.vocabulario:
            self.tfidf.ajustar(self.df["tokens"].tolist())

        self.vectores = self.tfidf.transformar(self.df["tokens"]).tolist()

    def _vectorizar_consulta(self, consulta):
        """Limpia y vectoriza la consulta como si fuera un versículo más.

        Parámetros
        consulta : str
            Frase del usuario o texto de un versículo.

        Retorna
        dict
            Vector TF-IDF de la consulta.
        """
        texto = self.preprocesador.limpiar_texto(consulta)
        tokens = self.preprocesador.quitar_stopwords(self.preprocesador.tokenizar(texto))
        return self.tfidf.calcularTF_IDF_Documento(tokens)

    def buscar(self, consulta, k = 5):
        """Devuelve los K versículos más parecidos a la consulta.

        Parámetros
        consulta : str
            Frase del usuario o texto de un versículo.
        k : int
            Cuántos versículos devolver (por defecto 5).

        Retorna
        pandas.DataFrame
            nombre_libro, numero_capitulo, numero_versiculo, texto_original y
            similitud, ordenado de mayor a menor.
        """
        vector_consulta = self._vectorizar_consulta(consulta)

        columnas = ["nombre_libro", "numero_capitulo", "numero_versiculo",
                    "texto_original", "similitud"]
        if not vector_consulta:
            return self.df.head(0).assign(similitud = [])[columnas]

        similitudes = [self.similitud.similitud_coseno(vector_consulta, vector)
                       for vector in self.vectores]

        resultado = self.df.copy()
        resultado["similitud"] = similitudes
        return resultado.nlargest(k, "similitud")[columnas].reset_index(drop = True)
