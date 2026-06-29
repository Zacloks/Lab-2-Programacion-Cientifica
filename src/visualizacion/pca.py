from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.preprocesamiento.analizador_vocabulario import AnalizadorVocabulario


class VisualizacionPCA:
    """Visualización bidimensional de los versículos mediante PCA.
    
    Toma cada versículo, lo representa como vector TF-IDF, y aplica PCA para
    comprimir esa representación de alta dimensión a solo dos componentes.
    Así puede graficar cada versículo como un punto en un plano, y ver si
    aparecen agrupaciones naturales por libro o testamento.

    Como trabajar con el vocabulario completo (miles de palabras) sería pesado,
    se limita a las top_n palabras más frecuentes del corpus para construir la
    matriz.
    """

    def __init__(self, df, tfidf, top_n=300):
        """Prepara la matriz TF-IDF densa y calcula las componentes principales.

        Parámetros
        df : pandas.DataFrame
            Corpus ya preprocesado, con la columna tokens y las columnas de
            categoría (testamento, nombre_genero, nombre_libro).
        tfidf : VectorTF_IDF
            Instancia del TF-IDF propio; se ajusta aquí sobre el corpus.
        top_n : int
            Cantidad de palabras más frecuentes a usar como dimensiones.
        """
        self.df = df
        self.tfidf = tfidf
        self.top_n = top_n
        self.coordenadas = None
        self.varianza = None

        self._construir_y_reducir()

    def _palabras_top(self):
        """Obtiene las top_n palabras más frecuentes del corpus.

        Retorna
        list[str]
            Las palabras que formarán las columnas de la matriz, ordenadas de
            mayor a menor frecuencia.
        """
        analizador = AnalizadorVocabulario()
        analizador.calcular_frecuencias(self.df)
        return list(analizador.palabras_mas_comunes(self.top_n)["palabra"])

    def _construir_matriz(self, palabras_top):
        """Arma la matriz densa de versículos por palabras top.

        Ajusta el TF-IDF sobre el corpus y, para cada versículo, llena una fila
        con los pesos TF-IDF de las palabras top (cero donde la palabra no
        aparece en ese versículo).

        Parámetros
        palabras_top : list[str]
            Palabras que definen las columnas de la matriz.

        Retorna
        numpy.ndarray
            Matriz de forma (cantidad de versículos, top_n).
        """
        indice_columna = {
            palabra: i for i, palabra in enumerate(palabras_top)
        }

        self.tfidf.ajustar(self.df["tokens"].tolist())

        matriz = np.zeros((len(self.df), len(palabras_top)))

        for fila, tokens in enumerate(self.df["tokens"]):
            vector = self.tfidf.calcularTF_IDF_Documento(tokens)

            for palabra, peso in vector.items():
                if palabra in indice_columna:
                    matriz[fila, indice_columna[palabra]] = peso

        return matriz

    def _construir_y_reducir(self):
        """Construye la matriz y aplica PCA para dejar dos componentes.

        Guarda en coordenadas el arreglo (versículos, 2) con PC1 y PC2, y en
        varianza la proporción de varianza explicada por cada componente.
        """
        palabras_top = self._palabras_top()
        matriz = self._construir_matriz(palabras_top)

        pca = PCA(n_components=2)
        self.coordenadas = pca.fit_transform(matriz)
        self.varianza = pca.explained_variance_ratio_

    def varianza_explicada(self):
        """Devuelve el porcentaje de varianza que captura cada componente.

        Retorna
        tuple[float, float]
            Porcentaje explicado por PC1 y por PC2.
        """
        return self.varianza[0] * 100, self.varianza[1] * 100

    def graficar(self, categoria="testamento"):
        """Grafica los versículos en el plano de las dos componentes.

        Cada punto es un versículo; el eje x es la primera componente principal
        y el eje y la segunda. Los puntos se colorean según la categoría
        indicada para poder distinguir agrupaciones.

        Parámetros
        categoria : str
            Columna del DataFrame usada para colorear (por defecto testamento;
            también sirve nombre_genero o nombre_libro).
        """
        valores = self.df[categoria].to_numpy()
        pc1_pct, pc2_pct = self.varianza_explicada()

        plt.figure(figsize=(11, 8))

        for valor in sorted(set(valores)):
            mascara = valores == valor
            plt.scatter(self.coordenadas[mascara, 0], self.coordenadas[mascara, 1], s=6, alpha=0.4, label=str(valor))

        plt.title(f"Versículos en el espacio PCA (coloreados por {categoria})")
        plt.xlabel(f"Componente principal 1 ({pc1_pct:.1f}% varianza)")
        plt.ylabel(f"Componente principal 2 ({pc2_pct:.1f}% varianza)")
        plt.legend(title=categoria, markerscale=2)
        plt.tight_layout()

        ruta = Path("resultados")
        ruta.mkdir(exist_ok=True)

        plt.savefig(ruta / f"pca_{categoria}.png", bbox_inches="tight")
        plt.close()