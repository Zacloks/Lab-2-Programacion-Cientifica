import pandas as pd
from pathlib import Path

class CargadorDatos:
    def __init__(self):
        self.ruta = Path(__file__).resolve().parent.parent / "data"

    def modificar_nombres_columnas(self, df, renombres):
        return df.rename(columns = renombres)

    def cargar_biblia(self):
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