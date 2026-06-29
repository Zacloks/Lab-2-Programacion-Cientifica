from src.preprocesamiento.preprocesador_texto import PreprocesadorTexto

class BuscadorSemantico:
    """Motor de búsqueda semántico sobre el corpus bíblico.

    Dada una frase escrita por el usuario o un versículo del dataset, encuentra
    los K versículos más parecidos del corpus. Para eso representa cada
    versículo con su vector TF-IDF y mide el parecido con la similitud del
    coseno propia.

    Al construirse precomputa los vectores TF-IDF de todos los versículos una
    sola vez, de modo que cada búsqueda solo tiene que vectorizar la consulta y
    compararla contra esos vectores.
    """

    def __init__(self, df, tfidf, similitud, preprocesador = None):
        """Prepara el buscador ajustando el TF-IDF y vectorizando el corpus.

        Parámetros
        df : pandas.DataFrame
            Corpus ya preprocesado, con la columna tokens y las columnas
            nombre_libro, numero_capitulo, numero_versiculo y texto_original.
        tfidf : VectorTF_IDF
            TF-IDF propio. Si todavía no está ajustado, se ajusta aquí.
        similitud : Similitud
            Objeto que calcula la similitud del coseno.
        preprocesador : PreprocesadorTexto o None
            Pipeline para limpiar la consulta. Si es None se crea uno nuevo, de
            modo que la consulta se procese igual que el corpus.
        """
        self.df = df.reset_index(drop = True)
        self.tfidf = tfidf
        self.similitud = similitud
        self.preprocesador = preprocesador if preprocesador is not None else PreprocesadorTexto()

        if not self.tfidf.vocabulario:
            self.tfidf.ajustar(self.df["tokens"].tolist())

        self.vectores = self.tfidf.transformar(self.df["tokens"]).tolist()

    def _vectorizar_consulta(self, consulta):
        """Limpia y vectoriza la consulta con el mismo pipeline del corpus.

        Parámetros
        consulta : str
            Frase del usuario o texto de un versículo.

        Retorna
        dict
            Vector TF-IDF disperso de la consulta (palabra a peso).
        """
        texto = self.preprocesador.limpiar_texto(consulta)
        tokens = self.preprocesador.quitar_stopwords(self.preprocesador.tokenizar(texto))
        return self.tfidf.calcularTF_IDF_Documento(tokens)

    def buscar(self, consulta, k = 5):
        """Devuelve los K versículos más similares a la consulta.

        Vectoriza la consulta, calcula su similitud del coseno contra todos los
        versículos del corpus y devuelve los K de mayor similitud.

        Parámetros
        consulta : str
            Frase del usuario o texto de un versículo del dataset.
        k : int
            Cantidad de versículos a devolver (por defecto 5).

        Retorna
        pandas.DataFrame
            Ranking con nombre_libro, numero_capitulo, numero_versiculo,
            texto_original y similitud, ordenado de mayor a menor similitud.
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
