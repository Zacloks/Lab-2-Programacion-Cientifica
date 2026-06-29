import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet = True)

class AnalizadorSentimiento:
    """Análisis de sentimiento de versículos mediante VADER.

    Implementa el requerimiento 3.7 del laboratorio: asigna un puntaje de
    sentimiento a cada versículo y luego agrega esos puntajes por libro y por
    capítulo para estudiar cómo varía el tono emocional a lo largo del corpus.

    Se utiliza VADER (Valence Aware Dictionary and sEntiment Reasoner), una
    técnica de análisis de sentimiento basada en un léxico con reglas, incluida
    en NLTK. VADER devuelve un puntaje compound en el rango [-1, 1], donde los
    valores cercanos a -1 indican un tono muy negativo, los cercanos a +1 un
    tono muy positivo y los cercanos a 0 un tono neutral.

    Decisión de diseño importante: el sentimiento se calcula sobre el texto
    original (sin preprocesar) y no sobre la columna ya limpia. VADER aprovecha
    las mayúsculas, los signos de puntuación y palabras vacías como "not" o "no"
    para detectar negaciones e intensidad, justo lo que el preprocesamiento
    elimina. Puntuar el texto limpio degradaría la calidad del análisis.
    """

    UMBRAL_NEUTRAL = 0.05

    def __init__(self):
        """Inicializa el analizador de intensidad de sentimiento de VADER."""
        self.analizador = SentimentIntensityAnalyzer()

    def puntuar_texto(self, texto):
        """Calcula el puntaje compound de sentimiento de un texto.

        Parámetros
        texto : str
            Texto de un versículo, idealmente el texto original sin preprocesar.

        Retorna
        float
            Puntaje compound en el rango [-1, 1].
        """
        return self.analizador.polarity_scores(str(texto))["compound"]

    def clasificar(self, puntaje):
        """Traduce un puntaje compound a una etiqueta cualitativa.

        Usa el umbral estándar recomendado por los autores de VADER: se
        considera positivo a partir de +0.05, negativo hasta -0.05 y neutral
        en el rango intermedio.

        Parámetros
        puntaje : float
            Puntaje compound de un versículo.

        Retorna
        str
            Una de las etiquetas "positivo", "negativo" o "neutral".
        """
        if puntaje >= self.UMBRAL_NEUTRAL:
            return "positivo"
        if puntaje <= -self.UMBRAL_NEUTRAL:
            return "negativo"
        return "neutral"

    def analizar(self, df, columna_texto = "texto_original"):
        """Asigna un puntaje y una etiqueta de sentimiento a cada versículo.

        Agrega dos columnas al DataFrame: sentimiento, con el puntaje compound,
        y etiqueta_sentimiento, con su clasificación cualitativa.

        Parámetros
        df : pandas.DataFrame
            Corpus con la columna de texto a puntuar. Se espera que incluya
            texto_original, ya que VADER funciona mejor sobre el texto crudo.
        columna_texto : str
            Nombre de la columna sobre la que se calcula el sentimiento. Por
            defecto texto_original.

        Retorna
        pandas.DataFrame
            El mismo DataFrame con las columnas sentimiento y
            etiqueta_sentimiento agregadas.
        """
        df["sentimiento"] = df[columna_texto].apply(self.puntuar_texto)
        df["etiqueta_sentimiento"] = df["sentimiento"].apply(self.clasificar)
        return df

    def agregar_por_libro(self, df):
        """Calcula el sentimiento promedio de cada libro del corpus.

        Conserva el orden canónico de los libros usando id_libro, de modo que
        el resultado puede graficarse como una evolución desde el primer libro
        hasta el último.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado con la columna sentimiento.

        Retorna
        pandas.DataFrame
            Tabla con id_libro, nombre_libro, testamento, sentimiento promedio
            y cantidad de versículos por libro, ordenada por id_libro.
        """
        return (
            df.groupby(["id_libro", "nombre_libro", "testamento"], as_index = False)
            .agg(sentimiento = ("sentimiento", "mean"),
                 versiculos = ("sentimiento", "size"))
            .sort_values("id_libro")
            .reset_index(drop = True)
        )

    def agregar_por_capitulo(self, df):
        """Calcula el sentimiento promedio de cada capítulo del corpus.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado con la columna sentimiento.

        Retorna
        pandas.DataFrame
            Tabla con id_libro, nombre_libro, numero_capitulo y sentimiento
            promedio, ordenada por libro y capítulo.
        """
        return (
            df.groupby(["id_libro", "nombre_libro", "numero_capitulo"], as_index = False)
            .agg(sentimiento = ("sentimiento", "mean"),
                 versiculos = ("sentimiento", "size"))
            .sort_values(["id_libro", "numero_capitulo"])
            .reset_index(drop = True)
        )

    def libros_extremos(self, df, n = 5):
        """Identifica los libros con sentimiento promedio más extremo.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado con la columna sentimiento.
        n : int
            Cantidad de libros a retornar en cada extremo (por defecto 5).

        Retorna
        tuple[pandas.DataFrame, pandas.DataFrame]
            Una dupla (mas_positivos, mas_negativos) con los n libros de mayor
            y menor sentimiento promedio respectivamente.
        """
        por_libro = self.agregar_por_libro(df)
        mas_positivos = por_libro.nlargest(n, "sentimiento").reset_index(drop = True)
        mas_negativos = por_libro.nsmallest(n, "sentimiento").reset_index(drop = True)
        return mas_positivos, mas_negativos

    def capitulos_extremos(self, df, n = 5):
        """Identifica los capítulos con sentimiento promedio más extremo.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado con la columna sentimiento.
        n : int
            Cantidad de capítulos a retornar en cada extremo (por defecto 5).

        Retorna
        tuple[pandas.DataFrame, pandas.DataFrame]
            Una dupla (mas_positivos, mas_negativos) con los n capítulos de
            mayor y menor sentimiento promedio respectivamente.
        """
        por_capitulo = self.agregar_por_capitulo(df)
        mas_positivos = por_capitulo.nlargest(n, "sentimiento").reset_index(drop = True)
        mas_negativos = por_capitulo.nsmallest(n, "sentimiento").reset_index(drop = True)
        return mas_positivos, mas_negativos

    def distribucion_etiquetas(self, df):
        """Cuenta cuántos versículos hay de cada etiqueta de sentimiento.

        Parámetros
        df : pandas.DataFrame
            Corpus ya puntuado con la columna etiqueta_sentimiento.

        Retorna
        pandas.Series
            Conteo de versículos por etiqueta (positivo, neutral, negativo).
        """
        return df["etiqueta_sentimiento"].value_counts()

    def comparar_preprocesamiento(self, df):
        """Compara el sentimiento medido sobre texto original versus limpio.

        Sirve para el análisis del requerimiento 3.8: evidencia cómo el
        preprocesamiento afecta los resultados. Calcula el puntaje sobre ambas
        versiones del texto y reporta la diferencia promedio, mostrando por qué
        conviene usar el texto original para el análisis de sentimiento.

        Parámetros
        df : pandas.DataFrame
            Corpus con las columnas texto_original y texto (ya limpia).

        Retorna
        float
            La diferencia absoluta media entre el puntaje calculado sobre el
            texto original y el calculado sobre el texto ya limpio.
        """
        sentimiento_original = df["texto_original"].apply(self.puntuar_texto)
        sentimiento_limpio = df["texto"].apply(self.puntuar_texto)
        return (sentimiento_original - sentimiento_limpio).abs().mean()
