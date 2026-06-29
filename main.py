from src.cargador_datos import CargadorDatos
from src.preprocesador_texto import PreprocesadorTexto
from src.vector import VectorTF_IDF

def main():
    cargador = CargadorDatos()
    preprocesador = PreprocesadorTexto()
    df = cargador.cargar_biblia()
    df = preprocesador.procesar(df)
    print(df.head())

if __name__ == "__main__":
    main()