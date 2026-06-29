import math

class VectorTF_IDF:
    def __init__(self):
        self.vocabulario = {}
        self.idf = {}
        self.totalDocumentos = {}
        
    def ajustar(self, serie_tokens):
        self.totalDocumentos = len(serie_tokens)
        df_counts = {}
        
        indice = 0
        for tokens in serie_tokens:
            tokensUnicos = set(tokens)
            for token in tokensUnicos:
                if token not in self.vocabulario:
                    self.vocabulario[token] =  indice
                df_counts[token] = df_counts.get(token, 0) + 1
                
        for token, df in df_counts.items():
            self.idf[token] = math.log(self.totalDocumentos / df)
            
    def calcularTF_IDF_Documento(self, tokens):
        tf_Documento = {}
        totalTerminos = len(tokens)
        
        if totalTerminos == 0:
            return {}
        
        for token in tokens:
            tf_Documento[token] = tf_Documento.get(token, 0) + 1
            
        tfIdf_Documento = {}
        for token, frecuencia in tf_Documento.items():
            if token in self.vocabulario:
                tf = frecuencia / totalTerminos
                tfIdf_Documento[token] = tf * self.idf[token]
                
        return tfIdf_Documento
    
    def transformar(self, serie_tokens):
        return serie_tokens.apply(self.calcularTF_IDF_Documento)