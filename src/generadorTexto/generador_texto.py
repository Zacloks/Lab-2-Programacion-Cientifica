import random

class ModeloNGramas:
    def __init__(self, n):
        self.n = n
        self.ngrams = {}
        self.vocabulario_unigrama = {}
        
    def entrenar(self, serie_tokens): #Entrenamiento de cada modelo
        print(f"Entrenando modelo {self.n}-gram...")
        
        for tokens in serie_tokens:
            if len(tokens) == 0:
                continue
            
            tokensExtendidos = ["<START>"] * max(1, self.n - 1) + tokens + ["<END>"]
            
            #Lógica para Unigramas
            if self.n == 1:
                for token in tokensExtendidos:
                    self.vocabulario_unigrama[token] = self.vocabulario_unigrama.get(token, 0) + 1
                    
            #Lógica para N-Gramas
            for i in range(len(tokensExtendidos) - self.n + 1):
                contexto = tuple(tokensExtendidos[i : i + self.n - 1])
                siguientePalabra = tokensExtendidos[i + self.n - 1]
                
                if contexto not in self.ngrams:
                    self.ngrams[contexto] = {}
                    
                self.ngrams[contexto][siguientePalabra] = self.ngrams[contexto].get(siguientePalabra, 0) + 1
                
    def generarTexto(self, palabraInicial = "<START>", longitudMax = 15):
        textoGenerado = []
        
        #Generación con Unigramas
        if self.n == 1:
            palabras = list(self.vocabulario_unigrama.keys())
            pesos = list(self.vocabulario_unigrama.values())
            
            for _ in range(longitudMax):
                siguiente = random.choices(palabras, weights=pesos, k = 1)[0]
                if siguiente == "<END>":
                    break
                if siguiente != "<START>":
                    textoGenerado.append(siguiente)
            return " ".join(textoGenerado)
        
        #Generación con N-Gramas
        contextoActual = ["<START>"] * (self.n - 2) + [palabraInicial]
        if palabraInicial != "<START>":
            textoGenerado.append(palabraInicial)
            
        for _ in range(longitudMax - len(textoGenerado)):
            contextoTupla = tuple(contextoActual)
            
            if contextoTupla in self.ngrams:
                opciones = self.ngrams[contextoTupla]
                palabrasSiguientes = list(opciones.keys())
                frecuencias = list(opciones.values())
                
                siguiente = random.choices(palabrasSiguientes, weights=frecuencias, k = 1)[0]
            
            else:
                break
            
            if siguiente == "<END>":
                break
            if siguiente != "<START>":
                textoGenerado.append(siguiente)
                
            contextoActual.pop(0)
            contextoActual.append(siguiente)
            
        return " ".join(textoGenerado)