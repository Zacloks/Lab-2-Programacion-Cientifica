from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from src.representacion.tfidf import VectorTF_IDF
from src.representacion.similitud import SimilitudCoseno

class VisualizacionYExplorador:
    """Visualizaciones del análisis exploratorio del corpus.

    Recibe el corpus ya preprocesado (con id_libro, nombre_libro y tokens) y
    guarda los gráficos en resultados/.
    """

    def __init__(self, df):
        """Guarda el corpus sobre el que se grafica.

        Parámetros
        df : pandas.DataFrame
            Corpus ya preprocesado.
        """
        self.df = df

    def graficoVersiculos_por_Libro(self):
        """Grafica en barras cuántos versículos tiene cada libro."""
        plt.figure(figsize = (15, 6))
        conteo = self.df["nombre_libro"].value_counts()
        sbn.barplot(x = conteo.index, y = conteo.values, hue = conteo.index, palette = "viridis", legend = False)
        plt.xticks(rotation = 90)
        plt.title("Cantidad de Versículos por Libro")
        plt.xlabel("Libro")
        plt.ylabel("Número de Versículos")
        Path("resultados").mkdir(exist_ok=True)
        plt.savefig("resultados/versiculos_por_libro.png", bbox_inches = "tight")
        plt.close()

    def graficoLongitud_Versiculos(self):
        """Grafica la distribución de la longitud de los versículos en tokens."""
        longitudes = self.df["tokens"].apply(len)

        plt.figure(figsize = (10, 5))
        sbn.histplot(longitudes, bins = 50, kde = True, color = "skyblue")
        plt.title("Distribución de la longitud de los Versículos (en tokens)")
        plt.xlabel("Cantidad de Tokens")
        plt.ylabel("Frecuencia")
        Path("resultados").mkdir(exist_ok=True)
        plt.savefig("resultados/longitud_versiculos.png", bbox_inches = "tight")
        plt.close()

    def _documentos_por_libro(self):
        """Junta los tokens de todos los versículos de cada libro en un documento.

        Concatena los tokens por libro respetando el orden canónico (id_libro)
        para poder comparar libros entre sí.

        Retorna
        pandas.DataFrame
            Tabla con id_libro, nombre_libro y los tokens del libro, ordenada
            por id_libro.
        """
        return (
            self.df.groupby(["id_libro", "nombre_libro"], sort = True)["tokens"]
            .agg(lambda series: [token for tokens in series for token in tokens])
            .reset_index()
            .sort_values("id_libro")
            .reset_index(drop = True)
        )

    def heatmap_similitud_libros(self):
        """Grafica el heatmap NxN de similitud del coseno entre libros.

        Representa cada libro como un documento TF-IDF y calcula la similitud
        entre todos los pares con SimilitudCoseno.
        """
        libros = self._documentos_por_libro()

        tfidf = VectorTF_IDF()
        tfidf.ajustar(libros["tokens"])
        vectores = list(tfidf.transformar(libros["tokens"]))

        matriz = SimilitudCoseno().matriz_similitud(vectores)
        nombres = libros["nombre_libro"].tolist()
        matriz_df = pd.DataFrame(matriz, index = nombres, columns = nombres)

        plt.figure(figsize = (16, 14))
        sbn.heatmap(matriz_df, cmap = "magma", square = True,
                    xticklabels = True, yticklabels = True,
                    cbar_kws = {"label": "Similitud del coseno"})
        plt.title("Similitud del coseno entre libros (TF-IDF por libro)")
        plt.xlabel("Libro")
        plt.ylabel("Libro")
        plt.tight_layout()
        Path("resultados").mkdir(exist_ok = True)
        plt.savefig("resultados/heatmap_similitud_libros.png", bbox_inches = "tight")
        plt.close()

    def grafico_palabras_frecuentes(self, n = 20):
        """Grafica las n palabras más frecuentes del corpus.

        Parámetros
        n : int
            Cuántas palabras mostrar (por defecto 20).
        """
        contador = Counter(token for tokens in self.df["tokens"] for token in tokens)
        comunes = contador.most_common(n)
        palabras = [palabra for palabra, _ in comunes]
        frecuencias = [frecuencia for _, frecuencia in comunes]

        plt.figure(figsize = (12, 7))
        sbn.barplot(x = frecuencias, y = palabras, hue = palabras,
                    palette = "rocket", legend = False)
        plt.title(f"Top {n} palabras más frecuentes del corpus")
        plt.xlabel("Frecuencia")
        plt.ylabel("Palabra")
        plt.tight_layout()
        Path("resultados").mkdir(exist_ok = True)
        plt.savefig("resultados/palabras_frecuentes.png", bbox_inches = "tight")
        plt.close()
