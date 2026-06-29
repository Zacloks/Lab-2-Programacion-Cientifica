import matplotlib.pyplot as plt
import seaborn as sbn
import pandas
import math

class VIsualizacionYExplorador:
    def __init__(self, df):
        self.df = df
        
    def graficoVersiculos_por_Libro(self):
        plt.figure(figsize = (15, 6))
        conteo = self.df["nombre_libro"].value_counts()
        sbn.barplot(x = conteo.index, y = conteo.values, palette = "viridis")
        plt.xticks(rotation = 90)
        plt.title("Cantidad de Versículos por Libro")
        plt.xlabel("Libro")
        plt.ylabel("Número de Versículos")
        plt.show()
        
    def graficoLongitud_Versiculos(self):
        longitudes = self.df["tokens"].apply(len)
        
        plt.figure(figsize = (10, 5))
        sbn.histplot(longitudes, bins = 50, kde = True, color = "skyblue")
        plt.title("Distribución de la longitud de los Versículos (en tokens)")
        plt.xlabel("Cantidad de Tokens")
        plt.ylabel("Frecuencia")
        plt.show()
        
    #FALTA IMPLEMENAT 1 MAS A ELECCION COMO MINIMO Y EL HEATMAP QUE ES OBLIGATORIO