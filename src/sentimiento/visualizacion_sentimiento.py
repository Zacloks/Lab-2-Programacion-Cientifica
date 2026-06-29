import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

class VisualizacionSentimiento:
    """Visualizaciones del análisis de sentimiento del corpus.

    Acompaña a AnalizadorSentimiento para cubrir el requerimiento 3.7:
    representa gráficamente la evolución del tono emocional a lo largo de los
    libros y capítulos, la distribución de los puntajes y los libros con
    sentimiento más extremo.

    Recibe el corpus ya puntuado (con la columna sentimiento) y una instancia
    de AnalizadorSentimiento, cuyas agregaciones reutiliza para no duplicar la
    lógica de cálculo.
    """

    def __init__(self, df, analizador):
        """Guarda el corpus puntuado y el analizador con el que agregar datos.

        Parámetros
        df : pandas.DataFrame
            Corpus ya procesado por AnalizadorSentimiento.analizar.
        analizador : AnalizadorSentimiento
            Analizador usado para obtener las agregaciones por libro y capítulo.
        """
        self.df = df
        self.analizador = analizador

    def distribucion_sentimiento(self):
        """Grafica la distribución de los puntajes de sentimiento por versículo.

        Permite ver qué tan polarizado está el corpus y cuántos versículos caen
        en la zona neutral cercana a cero.
        """
        plt.figure(figsize = (10, 5))
        sbn.histplot(self.df["sentimiento"], bins = 50, kde = True, color = "mediumpurple")
        plt.axvline(0, color = "black", linestyle = "--", linewidth = 1)
        plt.title("Distribución del sentimiento por versículo (VADER compound)")
        plt.xlabel("Puntaje de sentimiento")
        plt.ylabel("Cantidad de versículos")
        plt.tight_layout()
        plt.show()

    def evolucion_por_libro(self):
        """Grafica el sentimiento promedio de cada libro en orden canónico.

        Cada barra es un libro, ordenado desde Génesis hasta el último libro, y
        se colorea según el testamento. La línea horizontal en cero separa el
        tono positivo del negativo, mostrando la evolución emocional global.
        """
        por_libro = self.analizador.agregar_por_libro(self.df)

        plt.figure(figsize = (16, 6))
        sbn.barplot(data = por_libro, x = "nombre_libro", y = "sentimiento",
                    hue = "testamento", dodge = False, palette = "coolwarm")
        plt.axhline(0, color = "black", linewidth = 1)
        plt.xticks(rotation = 90)
        plt.title("Sentimiento promedio por libro (orden canónico)")
        plt.xlabel("Libro")
        plt.ylabel("Sentimiento promedio")
        plt.legend(title = "Testamento")
        plt.tight_layout()
        plt.show()

    def evolucion_por_capitulo(self, nombre_libro):
        """Grafica la evolución del sentimiento capítulo a capítulo de un libro.

        Muestra el arco emocional interno de un libro concreto, útil para libros
        con una progresión narrativa marcada.

        Parámetros
        nombre_libro : str
            Nombre del libro a graficar, tal como aparece en nombre_libro.
        """
        por_capitulo = self.analizador.agregar_por_capitulo(self.df)
        datos_libro = por_capitulo[por_capitulo["nombre_libro"] == nombre_libro]

        if datos_libro.empty:
            print(f"No se encontró el libro '{nombre_libro}' en el corpus.")
            return

        plt.figure(figsize = (12, 5))
        sbn.lineplot(data = datos_libro, x = "numero_capitulo", y = "sentimiento",
                     marker = "o", color = "teal")
        plt.axhline(0, color = "black", linestyle = "--", linewidth = 1)
        plt.title(f"Evolución del sentimiento en {nombre_libro} (por capítulo)")
        plt.xlabel("Capítulo")
        plt.ylabel("Sentimiento promedio")
        plt.tight_layout()
        plt.show()

    def libros_extremos(self, n = 10):
        """Grafica los libros con sentimiento promedio más positivo y negativo.

        Parámetros
        n : int
            Cantidad de libros a mostrar en cada extremo (por defecto 10).
        """
        mas_positivos, mas_negativos = self.analizador.libros_extremos(self.df, n)
        extremos = (
            pd.concat([mas_negativos, mas_positivos])
            .drop_duplicates("id_libro")
            .sort_values("sentimiento")
            .reset_index(drop = True)
        )

        plt.figure(figsize = (10, 8))
        colores = ["crimson" if valor < 0 else "seagreen" for valor in extremos["sentimiento"]]
        sbn.barplot(data = extremos, y = "nombre_libro", x = "sentimiento",
                    hue = "nombre_libro", palette = colores, legend = False)
        plt.axvline(0, color = "black", linewidth = 1)
        plt.title(f"Libros con sentimiento más extremo (top {n} de cada lado)")
        plt.xlabel("Sentimiento promedio")
        plt.ylabel("Libro")
        plt.tight_layout()
        plt.show()
