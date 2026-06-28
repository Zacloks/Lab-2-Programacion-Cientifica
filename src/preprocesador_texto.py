import re

class PreprocesadorTexto:
    def convertir_minusculas(self, texto):
        return texto.lower()
    
    def quitar_anotaciones(self, texto):
        return re.sub(r"\{.*?\}", "", texto)

    def eliminar_puntuacion_numeros(self, texto):
        return re.sub(r"[^a-zA-Z\s]", "", texto)

    def quitar_espacios_sobrantes(self, texto):
        return re.sub(r"\s+", " ", texto).strip()

    def limpiar_texto(self, texto):
        texto = self.quitar_anotaciones(texto)
        texto = self.convertir_minusculas(texto)
        texto = self.eliminar_puntuacion_numeros(texto)
        texto = self.quitar_espacios_sobrantes(texto)
        return texto

    def tokenizar(self, texto):
        return texto.split()

    def quitar_stopwords(self, tokens):
        stop_words = [] #Falta agregar stop words a la lista 
        return [token for token in tokens if token not in stop_words]

    def procesar(self, df):
        df["texto_original"] = df["texto"]
        df["texto"] = df["texto"].apply(self.limpiar_texto)
        df["tokens"] = df["texto"].apply(self.tokenizar).apply(self.quitar_stopwords)
        return df