import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet = True)

class AnalizadorSentimiento:
    """Análisis de sentimiento de versículos con VADER (3.7).

    Le pone un puntaje a cada versículo y luego agrega por libro y capítulo
    para ver cómo cambia el tono. VADER devuelve un compound entre -1
    (negativo) y +1 (positivo). Se calcula sobre el texto original, no el
    limpio, porque VADER usa las mayúsculas, la puntuación y palabras como
    "not" que el preprocesamiento borra.
    """

    UMBRAL_NEUTRAL = 0.05

    def __init__(self):
        """Inicializa el analizador de VADER."""
        self.analizador = SentimentIntensityAnalyzer()

    def puntuar_texto(self, texto):
        """Calcula el puntaje compound de un texto.

        Parámetros
        texto : str
            Texto del versículo, idealmente el original.

        Retorna
        float
            Puntaje compound entre -1 y 1.
        """
        return self.analizador.polarity_scores(str(texto))["compound"]

    def clasificar(self, puntaje):
        """Traduce un puntaje compound a una etiqueta.

        Usa el umbral estándar de VADER: positivo desde +0.05, negativo hasta
        -0.05 y neutral en el medio.

        Parámetros
        puntaje : float
            Puntaje compound del versículo.

        Retorna
        str
            "positivo", "negativo" o "neutral".
        """
        if puntaje >= self.UMBRAL_NEUTRAL:
            return "positivo"
        if puntaje <= -self.UMBRAL_NEUTRAL:
            return "negativo"
        return "neutral"

    def analizar(self, df, columna_texto = "texto_original"):
        """Le pone puntaje y etiqueta de sentimiento a cada versículo.

        Agrega las columnas sentimiento y etiqueta_sentimiento.

        Parámetros
        df : pandas.DataFrame
            Corpus con el texto a puntuar (idealmente texto_original).
        columna_texto : str
            Columna sobre la que se calcula (por defecto texto_original).

        Retorna
        pandas.DataFrame
            El mismo DataFrame con las dos columnas nuevas.
        """
        df["sentimiento"] = df[columna_texto].apply(self.puntuar_texto)
        df["etiqueta_sentimiento"] = df["sentimiento"].apply(self.clasificar)
        return df

    def agregar_por_libro(self, df):
        """Calcula el sentimiento promedio de cada libro.

        Mantiene el orden canónico con id_libro.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado.

        Retorna
        pandas.DataFrame
            id_libro, nombre_libro, testamento, sentimiento promedio y cantidad
            de versículos, ordenado por id_libro.
        """
        return (
            df.groupby(["id_libro", "nombre_libro", "testamento"], as_index = False)
            .agg(sentimiento = ("sentimiento", "mean"),
                 versiculos = ("sentimiento", "size"))
            .sort_values("id_libro")
            .reset_index(drop = True)
        )

    def agregar_por_capitulo(self, df):
        """Calcula el sentimiento promedio de cada capítulo.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado.

        Retorna
        pandas.DataFrame
            id_libro, nombre_libro, numero_capitulo y sentimiento promedio.
        """
        return (
            df.groupby(["id_libro", "nombre_libro", "numero_capitulo"], as_index = False)
            .agg(sentimiento = ("sentimiento", "mean"),
                 versiculos = ("sentimiento", "size"))
            .sort_values(["id_libro", "numero_capitulo"])
            .reset_index(drop = True)
        )

    def libros_extremos(self, df, n = 5):
        """Devuelve los libros de sentimiento más alto y más bajo.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado.
        n : int
            Cuántos libros por extremo (por defecto 5).

        Retorna
        tuple[pandas.DataFrame, pandas.DataFrame]
            (mas_positivos, mas_negativos).
        """
        por_libro = self.agregar_por_libro(df)
        mas_positivos = por_libro.nlargest(n, "sentimiento").reset_index(drop = True)
        mas_negativos = por_libro.nsmallest(n, "sentimiento").reset_index(drop = True)
        return mas_positivos, mas_negativos

    def capitulos_extremos(self, df, n = 5):
        """Devuelve los capítulos de sentimiento más alto y más bajo.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado.
        n : int
            Cuántos capítulos por extremo (por defecto 5).

        Retorna
        tuple[pandas.DataFrame, pandas.DataFrame]
            (mas_positivos, mas_negativos).
        """
        por_capitulo = self.agregar_por_capitulo(df)
        mas_positivos = por_capitulo.nlargest(n, "sentimiento").reset_index(drop = True)
        mas_negativos = por_capitulo.nsmallest(n, "sentimiento").reset_index(drop = True)
        return mas_positivos, mas_negativos

    def distribucion_etiquetas(self, df):
        """Cuenta cuántos versículos hay de cada etiqueta.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado.

        Retorna
        pandas.Series
            Conteo por etiqueta (positivo, neutral, negativo).
        """
        return df["etiqueta_sentimiento"].value_counts()

    def comparar_preprocesamiento(self, df):
        """Compara el sentimiento sobre el texto original y el limpio.

        Sirve para el análisis 3.8: muestra cuánto cambia el puntaje según el
        texto que se use.

        Parámetros
        df : pandas.DataFrame
            Corpus con texto_original y texto.

        Retorna
        float
            Diferencia absoluta media entre ambos puntajes.
        """
        sentimiento_original = df["texto_original"].apply(self.puntuar_texto)
        sentimiento_limpio = df["texto"].apply(self.puntuar_texto)
        return (sentimiento_original - sentimiento_limpio).abs().mean()
