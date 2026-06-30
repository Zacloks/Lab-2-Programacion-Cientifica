import random

class ModeloNGramas:
    """Generador de versículos falsos con un modelo de n-gramas.

    Aprende de las frecuencias de secuencias de palabras del corpus, sin
    modelos preentrenados ni redes. El parámetro n fija el orden (1 unigrama,
    2 bigrama, etc.). Usa los tokens <START> y <END> para marcar inicio y fin.
    """

    def __init__(self, n):
        """Inicializa el modelo para un orden n.

        Parámetros
        n : int
            Orden del modelo (1 unigrama, 2 bigrama, 3 trigrama...).
        """
        self.n = n
        self.ngrams = {}
        self.vocabulario_unigrama = {}

    def entrenar(self, serie_tokens):
        """Aprende las frecuencias de n-gramas del corpus.

        Para cada versículo cuenta, por cada contexto de n-1 palabras, qué
        palabra le sigue.

        Parámetros
        serie_tokens : Iterable[list[str]]
            Tokens de cada versículo.
        """
        print(f"Entrenando modelo {self.n}-gram...")

        for tokens in serie_tokens:
            if len(tokens) == 0:
                continue

            tokens_extendidos = ["<START>"] * max(1, self.n - 1) + tokens + ["<END>"]

            if self.n == 1:
                for token in tokens_extendidos:
                    self.vocabulario_unigrama[token] = self.vocabulario_unigrama.get(token, 0) + 1

            for i in range(len(tokens_extendidos) - self.n + 1):
                contexto = tuple(tokens_extendidos[i : i + self.n - 1])
                siguiente_palabra = tokens_extendidos[i + self.n - 1]

                if contexto not in self.ngrams:
                    self.ngrams[contexto] = {}

                self.ngrams[contexto][siguiente_palabra] = self.ngrams[contexto].get(siguiente_palabra, 0) + 1

    def generar_texto(self, palabra_inicial = "<START>", longitud_maxima = 15):
        """Genera un versículo falso desde una palabra inicial.

        Va eligiendo la siguiente palabra al azar según las frecuencias
        aprendidas. Se detiene en <END>, al llegar al largo máximo o si el
        contexto no tiene continuación.

        Parámetros
        palabra_inicial : str
            Palabra desde la que arranca (por defecto <START>).
        longitud_maxima : int
            Máximo de palabras a generar (por defecto 15).

        Retorna
        str
            El versículo generado.
        """
        texto_generado = []

        if self.n == 1:
            palabras = list(self.vocabulario_unigrama.keys())
            pesos = list(self.vocabulario_unigrama.values())

            for _ in range(longitud_maxima):
                siguiente = random.choices(palabras, weights = pesos, k = 1)[0]
                if siguiente == "<END>":
                    break
                if siguiente != "<START>":
                    texto_generado.append(siguiente)
            return " ".join(texto_generado)

        contexto_actual = ["<START>"] * (self.n - 2) + [palabra_inicial]
        if palabra_inicial != "<START>":
            texto_generado.append(palabra_inicial)

        for _ in range(longitud_maxima - len(texto_generado)):
            contexto_tupla = tuple(contexto_actual)

            if contexto_tupla in self.ngrams:
                opciones = self.ngrams[contexto_tupla]
                palabras_siguientes = list(opciones.keys())
                frecuencias = list(opciones.values())
                siguiente = random.choices(palabras_siguientes, weights = frecuencias, k = 1)[0]
            else:
                break

            if siguiente == "<END>":
                break
            if siguiente != "<START>":
                texto_generado.append(siguiente)

            contexto_actual.pop(0)
            contexto_actual.append(siguiente)

        return " ".join(texto_generado)