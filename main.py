from src.datos.cargador_datos import CargadorDatos
from src.preprocesamiento.preprocesador_texto import PreprocesadorTexto
from src.preprocesamiento.analizador_vocabulario import AnalizadorVocabulario
from src.representacion.tfidf import VectorTF_IDF
from src.visualizacion.exploratorio import VisualizacionYExplorador

def main():
    cargador = CargadorDatos()
    preprocesador = PreprocesadorTexto()
    df = cargador.cargar_biblia()
    df = preprocesador.procesar(df)
    print(df.head())

    analizador = AnalizadorVocabulario()
    analizador.construir_vocabulario(df)
    print(f"\nTamaño del vocabulario: {analizador.tamano_vocabulario()} palabras únicas")
    print("\nTop 15 palabras más frecuentes:")
    print(analizador.palabras_mas_comunes(15).to_string(index = False))

    explorador = VisualizacionYExplorador(df)
    explorador.graficoLongitud_Versiculos()
    explorador.graficoVersiculos_por_Libro()

if __name__ == "__main__":
    main()