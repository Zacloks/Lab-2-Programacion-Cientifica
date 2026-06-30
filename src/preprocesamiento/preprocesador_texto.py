import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet = True)

stopwords_extras = {
    "shall", "shalt", "unto", "thee", "thou", "thy", "thine", "ye",
    "hath", "doth", "art", "thus",
    "said", "saith", "saying", "say", "says",
    "come", "came", "go", "went", "made", "let",
    "one", "also", "upon", "yet", "us",
}

class PreprocesadorTexto:
    """Limpia y tokeniza los versículos del corpus.

    Pasa cada versículo a minúsculas, le quita puntuación, números y caracteres
    especiales, lo tokeniza y elimina las stopwords del inglés más una lista de
    dominio. El método de entrada es procesar.
    """

    def __init__(self):
        """Arma el conjunto de stopwords juntando las de NLTK con las extras."""

        self.stop_words = (
            set(stopwords.words("english")) |
            stopwords_extras
        )

    def convertir_minusculas(self, texto):
        """Pasa el texto a minúsculas."""
        return texto.lower()

    def quitar_anotaciones(self, texto):
        """Quita las anotaciones del traductor, que vienen entre llaves."""
        return re.sub(r"\{.*?\}", "", texto)

    def eliminar_puntuacion_numeros(self, texto):
        """Deja solo letras y espacios, quitando puntuación, números y símbolos."""
        return re.sub(r"[^a-zA-Z\s]", "", texto)

    def quitar_espacios_sobrantes(self, texto):
        """Junta los espacios múltiples en uno solo y recorta los extremos."""
        return re.sub(r"\s+", " ", texto).strip()

    def limpiar_texto(self, texto):
        """Aplica toda la limpieza a un texto.

        El orden importa: primero quita las anotaciones entre llaves y después
        la puntuación.

        Parámetros
        texto : str
            Texto original del versículo.

        Retorna
        str
            Texto en minúsculas, sin anotaciones, puntuación ni espacios de más.
        """
        texto = self.quitar_anotaciones(texto)
        texto = self.convertir_minusculas(texto)
        texto = self.eliminar_puntuacion_numeros(texto)
        texto = self.quitar_espacios_sobrantes(texto)
        return texto

    def tokenizar(self, texto):
        """Parte el texto limpio en una lista de palabras."""
        return texto.split()

    def quitar_stopwords(self, tokens):
        """Saca las stopwords de una lista de tokens.

        Parámetros
        tokens : list[str]
            Tokens del versículo.

        Retorna
        list[str]
            Los tokens que no son stopwords.
        """
        return [token for token in tokens if token not in self.stop_words]

    def procesar(self, df):
        """Aplica todo el pipeline de preprocesamiento al DataFrame.

        Guarda el texto original en texto_original, deja el texto limpio en
        texto y agrega la columna tokens.

        Parámetros
        df : pandas.DataFrame
            Corpus con una columna texto.

        Retorna
        pandas.DataFrame
            El mismo DataFrame con texto_original y tokens.
        """
        df["texto_original"] = df["texto"]
        df["texto"] = df["texto"].apply(self.limpiar_texto)
        df["tokens"] = df["texto"].apply(self.tokenizar).apply(self.quitar_stopwords)
        return df
