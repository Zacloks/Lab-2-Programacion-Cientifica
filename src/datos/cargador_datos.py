import pandas as pd
from pathlib import Path

class CargadorDatos:
    """Carga los CSV del corpus bíblico y los une en un solo DataFrame.

    Lee los versículos, el catálogo de libros y el de géneros, pasa los
    nombres de columnas a español y los junta por id_libro e id_genero.
    """

    def __init__(self):
        """Apunta a la carpeta data, que está en la raíz del proyecto."""
        self.ruta = Path(__file__).resolve().parent.parent.parent / "data"

    def modificar_nombres_columnas(self, df, renombres):
        """Renombra columnas de un DataFrame.

        Parámetros
        df : pandas.DataFrame
            DataFrame a modificar.
        renombres : dict
            Mapeo de nombre actual a nombre nuevo.

        Retorna
        pandas.DataFrame
            El DataFrame con las columnas renombradas.
        """
        return df.rename(columns = renombres)

    def cargar_biblia(self):
        """Carga los tres CSV y los une en un único DataFrame.

        Une los versículos con sus libros por id_libro y luego con los géneros
        por id_genero.

        Retorna
        pandas.DataFrame
            Un versículo por fila, con su texto y sus metadatos.
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
