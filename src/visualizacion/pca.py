from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.preprocesamiento.analizador_vocabulario import AnalizadorVocabulario

class VisualizacionPCA:
    """Visualización 2D de los versículos con PCA.

    Representa cada versículo con su vector TF-IDF y usa PCA para bajarlo a dos
    componentes, así puede graficar cada versículo como un punto y ver si se
    agrupan por libro o testamento. Para que la matriz no sea tan pesada usa
    solo las top_n palabras más frecuentes.
    """

    def __init__(self, df, tfidf, top_n=300):
        """Arma la matriz TF-IDF y calcula las dos componentes principales.

        Parámetros
        df : pandas.DataFrame
            Corpus preprocesado, con tokens y las columnas de categoría.
        tfidf : VectorTF_IDF
            TF-IDF propio; se ajusta aquí sobre el corpus.
        top_n : int
            Cuántas palabras frecuentes usar como dimensiones.
        """
        self.df = df
        self.tfidf = tfidf
        self.top_n = top_n
        self.coordenadas = None
        self.varianza = None

        self._construir_y_reducir()

    def _palabras_top(self):
        """Saca las top_n palabras más frecuentes del corpus.

        Retorna
        list[str]
            Las palabras que serán las columnas de la matriz.
        """
        analizador = AnalizadorVocabulario()
        analizador.calcular_frecuencias(self.df)
        return list(analizador.palabras_mas_comunes(self.top_n)["palabra"])

    def _construir_matriz(self, palabras_top):
        """Arma la matriz densa de versículos por palabras top.

        Llena cada fila con los pesos TF-IDF de las palabras top, con cero donde
        la palabra no aparece.

        Parámetros
        palabras_top : list[str]
            Palabras que definen las columnas.

        Retorna
        numpy.ndarray
            Matriz de forma (versículos, top_n).
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
        """Arma la matriz y le aplica PCA para dejar dos componentes.

        Guarda las coordenadas (versículos, 2) y la varianza explicada de cada
        componente.
        """
        palabras_top = self._palabras_top()
        matriz = self._construir_matriz(palabras_top)

        pca = PCA(n_components=2)
        self.coordenadas = pca.fit_transform(matriz)
        self.varianza = pca.explained_variance_ratio_

    def varianza_explicada(self):
        """Devuelve el porcentaje de varianza de cada componente.

        Retorna
        tuple[float, float]
            Porcentaje de PC1 y PC2.
        """
        return self.varianza[0] * 100, self.varianza[1] * 100

    def graficar(self, categoria="testamento"):
        """Grafica los versículos en el plano de las dos componentes.

        Cada punto es un versículo y los colores van por la categoría indicada.

        Parámetros
        categoria : str
            Columna usada para colorear (por defecto testamento; también sirve
            nombre_genero o nombre_libro).
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
