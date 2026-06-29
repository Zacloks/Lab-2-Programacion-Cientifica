from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

from src.representacion.tfidf import VectorTF_IDF
from src.representacion.similitud import SimilitudCoseno

class VisualizacionYExplorador:
    """Visualizaciones del análisis exploratorio del corpus.

    Genera gráficos que ayudan a entender la estructura y los patrones del
    texto bíblico ya preprocesado. Recibe el DataFrame del corpus, que debe
    incluir las columnas id_libro, nombre_libro y tokens, y produce las figuras
    con matplotlib y seaborn.
    """

    def __init__(self, df):
        """Guarda el DataFrame del corpus, ya preprocesado, sobre el que se grafica.

        Parámetros
        df : pandas.DataFrame
            Corpus ya preprocesado.
        """
        self.df = df

    def graficoVersiculos_por_Libro(self):
        """Grafica la cantidad de versículos que tiene cada libro.

        Produce un gráfico de barras ordenado según el conteo de versículos de
        la columna nombre_libro.
        """
        plt.figure(figsize = (15, 6))
        conteo = self.df["nombre_libro"].value_counts()
        sbn.barplot(x = conteo.index, y = conteo.values, palette = "viridis")
        plt.xticks(rotation = 90)
        plt.title("Cantidad de Versículos por Libro")
        plt.xlabel("Libro")
        plt.ylabel("Número de Versículos")
        plt.show()

    def graficoLongitud_Versiculos(self):
        """Grafica la distribución de la longitud de los versículos.

        Mide la longitud como la cantidad de tokens por versículo y muestra su
        histograma junto a una estimación de densidad (KDE).
        """
        longitudes = self.df["tokens"].apply(len)

        plt.figure(figsize = (10, 5))
        sbn.histplot(longitudes, bins = 50, kde = True, color = "skyblue")
        plt.title("Distribución de la longitud de los Versículos (en tokens)")
        plt.xlabel("Cantidad de Tokens")
        plt.ylabel("Frecuencia")
        plt.show()

    def _documentos_por_libro(self):
        """Agrupa los tokens de todos los versículos de cada libro en un documento.

        Para comparar libros entre sí hay que representar cada libro como una
        sola unidad de texto. Se concatenan los tokens de todos sus versículos,
        respetando el orden canónico (id_libro), de modo que el libro queda como
        un único documento sobre el que calcular TF-IDF.

        Retorna
        pandas.DataFrame
            Tabla con id_libro, nombre_libro y la lista de tokens del libro
            completo, ordenada por id_libro.
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

        Es la visualización obligatoria del laboratorio. Representa cada uno de
        los 66 libros como un documento TF-IDF (usando la implementación propia
        del equipo) y calcula la similitud del coseno entre todos los pares con
        el módulo SimilitudCoseno. El resultado es una matriz simétrica de 66x66
        donde los valores altos indican libros con vocabulario parecido.
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
        plt.show()

    def grafico_palabras_frecuentes(self, n = 20):
        """Grafica las n palabras más frecuentes del corpus ya preprocesado.

        Cuenta las apariciones de cada token en todo el corpus (texto limpio y
        sin stopwords) y muestra las más comunes en un gráfico de barras
        horizontal, ofreciendo una vista rápida de los términos que dominan el
        vocabulario.

        Parámetros
        n : int
            Cantidad de palabras a mostrar (por defecto 20).
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
        plt.show()