from collections import Counter
import pandas as pd

class AnalizadorVocabulario:
    """Construye el vocabulario y cuenta las frecuencias de palabras del corpus.

    Trabaja sobre la columna tokens de un DataFrame ya preprocesado.
    """

    def __init__(self):
        """Deja el vocabulario y la tabla de frecuencias vacíos."""
        self.vocabulario = {}
        self.frecuencias = pd.DataFrame(columns = ["palabra", "frecuencia"])

    def _contar_tokens(self, df):
        """Cuenta cuántas veces aparece cada palabra en todo el corpus.

        Parámetros
        df : pandas.DataFrame
            Corpus con la columna tokens.

        Retorna
        collections.Counter
            Conteo de palabra a número de apariciones.
        """
        contador = Counter()
        for tokens in df["tokens"]:
            contador.update(tokens)
        return contador

    def calcular_frecuencias(self, df):
        """Calcula la frecuencia de cada palabra del corpus.

        Parámetros
        df : pandas.DataFrame
            Corpus con la columna tokens.

        Retorna
        pandas.DataFrame
            Tabla con palabra y frecuencia, ordenada de mayor a menor.
        """
        contador = self._contar_tokens(df)
        self.frecuencias = (
            pd.DataFrame(contador.items(), columns = ["palabra", "frecuencia"])
            .sort_values("frecuencia", ascending = False)
            .reset_index(drop = True)
        )
        return self.frecuencias

    def construir_vocabulario(self, df):
        """Arma el vocabulario asignando un índice a cada palabra.

        Ordena por frecuencia, así la palabra más común queda con el índice 0.
        Si no hay frecuencias calculadas, las calcula antes.

        Parámetros
        df : pandas.DataFrame
            Corpus con la columna tokens.

        Retorna
        dict
            Mapeo de palabra a índice.
        """
        if self.frecuencias.empty:
            self.calcular_frecuencias(df)
        self.vocabulario = {
            palabra: indice
            for indice, palabra in enumerate(self.frecuencias["palabra"])
        }
        return self.vocabulario

    def tamano_vocabulario(self):
        """Devuelve cuántas palabras únicas tiene el vocabulario."""
        return len(self.vocabulario)

    def palabras_mas_comunes(self, n = 15):
        """Devuelve las n palabras más frecuentes.

        Parámetros
        n : int
            Cuántas palabras devolver (por defecto 15).

        Retorna
        pandas.DataFrame
            Las primeras n filas de la tabla de frecuencias.
        """
        return self.frecuencias.head(n)
