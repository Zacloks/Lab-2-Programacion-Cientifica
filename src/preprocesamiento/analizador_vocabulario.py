from collections import Counter
import pandas as pd

class AnalizadorVocabulario:
    """Etapas finales del preprocesamiento a nivel de corpus.

    Cubre los dos últimos pasos del pipeline que pide el laboratorio: la
    construcción del vocabulario y el cálculo de frecuencias de palabras.

    Opera sobre la columna tokens de un DataFrame que ya pasó por el
    PreprocesadorTexto, es decir, texto limpio, tokenizado y sin stopwords.
    """

    def __init__(self):
        """Inicializa el vocabulario y la tabla de frecuencias vacíos."""
        self.vocabulario = {}
        self.frecuencias = pd.DataFrame(columns = ["palabra", "frecuencia"])

    def _contar_tokens(self, df):
        """Cuenta las apariciones de cada palabra en todo el corpus.

        Es un método interno que recorre la columna tokens acumulando los
        conteos de todos los versículos en un único contador.

        Parámetros
        df : pandas.DataFrame
            DataFrame con una columna tokens (listas de palabras).

        Retorna
        collections.Counter
            Contador con la forma palabra a número de apariciones en el corpus.
        """
        contador = Counter()
        for tokens in df["tokens"]:
            contador.update(tokens)
        return contador

    def calcular_frecuencias(self, df):
        """Calcula la frecuencia global de cada palabra del corpus.

        Parámetros
        df : pandas.DataFrame
            DataFrame con una columna tokens.

        Retorna
        pandas.DataFrame
            Tabla con columnas palabra y frecuencia, ordenada de mayor a menor
            frecuencia. También queda guardada en el atributo frecuencias.
        """
        contador = self._contar_tokens(df)
        self.frecuencias = (
            pd.DataFrame(contador.items(), columns = ["palabra", "frecuencia"])
            .sort_values("frecuencia", ascending = False)
            .reset_index(drop = True)
        )
        return self.frecuencias

    def construir_vocabulario(self, df):
        """Construye el vocabulario del corpus a partir de las frecuencias.

        Asigna a cada palabra única un índice entero estable, ordenado por
        frecuencia descendente, de modo que la palabra más frecuente recibe el
        índice 0. Si las frecuencias aún no se calcularon, las calcula primero.

        Parámetros
        df : pandas.DataFrame
            DataFrame con una columna tokens.

        Retorna
        dict
            Mapeo de cada palabra a su índice. También queda guardado en el
            atributo vocabulario.
        """
        if self.frecuencias.empty:
            self.calcular_frecuencias(df)
        self.vocabulario = {
            palabra: indice
            for indice, palabra in enumerate(self.frecuencias["palabra"])
        }
        return self.vocabulario

    def tamano_vocabulario(self):
        """Devuelve la cantidad de palabras únicas del vocabulario."""
        return len(self.vocabulario)

    def palabras_mas_comunes(self, n = 15):
        """Devuelve las n palabras más frecuentes del corpus.

        Parámetros
        n : int
            Cantidad de palabras a retornar (por defecto 15).

        Retorna
        pandas.DataFrame
            Las primeras n filas de la tabla de frecuencias.
        """
        return self.frecuencias.head(n)
