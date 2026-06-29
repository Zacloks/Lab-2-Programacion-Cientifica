import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet = True)

class PreprocesadorTexto:
    """Pipeline de limpieza y tokenización de versículos.

    Aplica, versículo a versículo, las etapas básicas de preprocesamiento que
    pide el laboratorio: normalización a minúsculas, eliminación de puntuación,
    números y caracteres especiales, tokenización y eliminación de palabras
    vacías (stopwords) del inglés.

    El método de entrada es procesar, que recibe el DataFrame del corpus y
    devuelve el mismo DataFrame con las columnas texto_original, texto (ya
    limpio) y tokens (la lista de palabras sin stopwords).
    """

    def __init__(self):
        """Carga el conjunto de stopwords del inglés provisto por NLTK."""
        self.stop_words = set(stopwords.words("english"))

    def convertir_minusculas(self, texto):
        """Convierte todo el texto a minúsculas para unificar el vocabulario."""
        return texto.lower()

    def quitar_anotaciones(self, texto):
        """Elimina las anotaciones editoriales encerradas entre llaves.

        La versión WEB del dataset incluye notas del traductor entre llaves que
        no forman parte del texto bíblico y deben descartarse.
        """
        return re.sub(r"\{.*?\}", "", texto)

    def eliminar_puntuacion_numeros(self, texto):
        """Elimina todo carácter que no sea una letra o un espacio.

        Cubre en un solo paso la eliminación de signos de puntuación, números y
        caracteres especiales, dejando únicamente letras del alfabeto inglés.
        """
        return re.sub(r"[^a-zA-Z\s]", "", texto)

    def quitar_espacios_sobrantes(self, texto):
        """Colapsa los espacios múltiples en uno solo y recorta los extremos."""
        return re.sub(r"\s+", " ", texto).strip()

    def limpiar_texto(self, texto):
        """Aplica la secuencia completa de limpieza a un texto.

        El orden importa: primero se quitan las anotaciones entre llaves
        mientras las llaves aún existen, y recién después se elimina la
        puntuación restante.

        Parámetros
        texto : str
            Texto original de un versículo.

        Retorna
        str
            Texto en minúsculas, sin anotaciones, puntuación ni espacios extra.
        """
        texto = self.quitar_anotaciones(texto)
        texto = self.convertir_minusculas(texto)
        texto = self.eliminar_puntuacion_numeros(texto)
        texto = self.quitar_espacios_sobrantes(texto)
        return texto

    def tokenizar(self, texto):
        """Divide el texto limpio en una lista de palabras (tokens)."""
        return texto.split()

    def quitar_stopwords(self, tokens):
        """Filtra las palabras vacías (stopwords) de una lista de tokens.

        Parámetros
        tokens : list[str]
            Lista de tokens de un versículo.

        Retorna
        list[str]
            Los tokens que no son stopwords.
        """
        return [token for token in tokens if token not in self.stop_words]

    def procesar(self, df):
        """Aplica todo el pipeline de preprocesamiento sobre el DataFrame.

        Conserva el texto original en la columna texto_original, reemplaza la
        columna texto por su versión limpia y agrega la columna tokens con las
        palabras ya tokenizadas y sin stopwords.

        Parámetros
        df : pandas.DataFrame
            DataFrame del corpus con una columna texto.

        Retorna
        pandas.DataFrame
            El mismo DataFrame enriquecido con texto_original y tokens.
        """
        df["texto_original"] = df["texto"]
        df["texto"] = df["texto"].apply(self.limpiar_texto)
        df["tokens"] = df["texto"].apply(self.tokenizar).apply(self.quitar_stopwords)
        return df
