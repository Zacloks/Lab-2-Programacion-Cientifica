from src.cargador_datos import CargadorDatos
from src.preprocesador_texto import PreprocesadorTexto
from src.vector import VectorTF_IDF
from src.visualizacion_analisis import VIsualizacionYExplorador

def main():
    cargador = CargadorDatos()
    preprocesador = PreprocesadorTexto()
    df = cargador.cargar_biblia()
    df = preprocesador.procesar(df)
    print(df.head())
    
    explorador = VIsualizacionYExplorador(df)
    explorador.graficoLongitud_Versiculos()
    explorador.graficoVersiculos_por_Libro()

if __name__ == "__main__":
    main()