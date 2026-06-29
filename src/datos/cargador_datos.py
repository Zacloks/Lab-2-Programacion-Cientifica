import pandas as pd
from pathlib import Path

class CargadorDatos:
    """Carga y unifica los archivos CSV del corpus bíblico.

    El dataset está dividido en tres archivos relacionados: t_web.csv con los
    versículos (texto y su ubicación libro/capítulo/versículo), key_english.csv
    con el catálogo de libros (nombre, testamento, género) y
    key_genre_english.csv con el catálogo de géneros literarios.

    Esta clase los lee, normaliza los nombres de columnas a español y los une
    en un único DataFrame listo para el preprocesamiento.
    """

    def __init__(self):
        """Fija la ruta a la carpeta data ubicada en la raíz del proyecto.

        El archivo vive en src/datos/, por lo que se suben tres niveles para
        llegar a la raíz del repositorio y desde ahí a la carpeta data.
        """
        self.ruta = Path(__file__).resolve().parent.parent.parent / "data"

    def modificar_nombres_columnas(self, df, renombres):
        """Renombra las columnas de un DataFrame.

        Parámetros
        df : pandas.DataFrame
            DataFrame al que se le renombrarán columnas.
        renombres : dict
            Mapeo de nombre actual a nombre nuevo.

        Retorna
        pandas.DataFrame
            DataFrame con las columnas renombradas.
        """
        return df.rename(columns = renombres)

    def cargar_biblia(self):
        """Carga los tres CSV y los une en un único DataFrame.

        La unión se hace primero por id_libro, para juntar cada versículo con
        los datos de su libro, y luego por id_genero, para agregar el género
        literario. Así cada fila final corresponde a un versículo enriquecido
        con el nombre de su libro, su testamento y su género.

        Retorna
        pandas.DataFrame
            Un versículo por fila, con su texto y sus metadatos jerárquicos.
        """
        df_versiculos = pd.read_csv(self.ruta / "t_web.csv", encoding="latin-1")
        df_libros = pd.read_csv(self.ruta / "key_english.csv", encoding="latin-1")
        df_generos = pd.read_csv(self.ruta / "key_genre_english.csv", encoding="latin-1")

        nombres_columnas_versiculos = {
            "id": "id_versiculo",
            "b": "id_libro",
            "c": "numero_capitulo",
            "v": "numero_versiculo",
            "t": "texto"
        }

        nombres_columnas_libros = {
            "b": "id_libro",
            "n": "nombre_libro",
            "t": "testamento",
            "g": "id_genero"
        }

        nombres_columnas_generos = {
            "g": "id_genero",
            "n": "nombre_genero"
        }

        df_versiculos = self.modificar_nombres_columnas(df_versiculos, nombres_columnas_versiculos)
        df_libros = self.modificar_nombres_columnas(df_libros, nombres_columnas_libros)
        df_generos = self.modificar_nombres_columnas(df_generos, nombres_columnas_generos)

        df_biblia = pd.merge(df_versiculos, df_libros, on = "id_libro", how = "left")
        df_biblia = pd.merge(df_biblia, df_generos, on = "id_genero", how = "left")

        return df_biblia