import matplotlib.pyplot as plt
import seaborn as sbn

class VisualizacionYExplorador:
    """Visualizaciones del análisis exploratorio del corpus.

    Genera gráficos que ayudan a entender la estructura y los patrones del
    texto bíblico ya preprocesado. Recibe el DataFrame del corpus, que debe
    incluir las columnas nombre_libro y tokens, y produce las figuras con
    matplotlib y seaborn.

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

    # FALTA IMPLEMENTAR 1 MÁS A ELECCIÓN COMO MÍNIMO Y EL HEATMAP QUE ES OBLIGATORIO
