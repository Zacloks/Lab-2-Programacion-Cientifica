from src.datos.cargador_datos import CargadorDatos
from src.preprocesamiento.preprocesador_texto import PreprocesadorTexto
from src.preprocesamiento.analizador_vocabulario import AnalizadorVocabulario
from src.representacion.tfidf import VectorTF_IDF
from src.representacion.similitud import Similitud
from src.modelos.buscador_semantico import BuscadorSemantico
from src.visualizacion.exploratorio import VisualizacionYExplorador
from src.visualizacion.pca import VisualizacionPCA

def cargar_preprocesar():
    """Carga el corpus bíblico y le aplica el pipeline de preprocesamiento.

    Retorna
    pandas.DataFrame
        El corpus con las columnas texto_original, texto (limpio) y tokens.
    """
    print("\nEtapa 1: carga y preprocesamiento del corpus")
    df = CargadorDatos().cargar_biblia()
    df = PreprocesadorTexto().procesar(df)
    print(f"Versículos cargados: {len(df)}")
    print("\nPrimeras filas del corpus ya preprocesado:")
    print(df[["nombre_libro", "testamento", "texto", "tokens"]].head())
    return df


def analizar_vocabulario(df):
    """Construye el vocabulario y muestra las palabras más frecuentes.

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.

    Retorna
    AnalizadorVocabulario
        El analizador con el vocabulario y las frecuencias ya calculados.
    """
    print("\nEtapa 2: vocabulario y frecuencias de palabras")
    analizador = AnalizadorVocabulario()
    analizador.construir_vocabulario(df)
    print(f"Tamaño del vocabulario: {analizador.tamano_vocabulario()} palabras únicas")
    print("\nTop 15 palabras más frecuentes:")
    print(analizador.palabras_mas_comunes(15).to_string(index = False))
    return analizador


def demostrar_tfidf(df):
    """Ajusta el TF-IDF propio y muestra los términos con más peso de un versículo.

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 3: representación TF-IDF (implementación propia)")
    tfidf = VectorTF_IDF()
    tfidf.ajustar(df["tokens"].tolist())
    print(f"Documentos ajustados: {tfidf.totalDocumentos}")
    print(f"Palabras en el vocabulario TF-IDF: {len(tfidf.vocabulario)}")

    ejemplo = df.iloc[0]
    vector = tfidf.calcularTF_IDF_Documento(ejemplo["tokens"])
    pesos_ordenados = sorted(vector.items(), key = lambda par: par[1], reverse = True)

    print(f"\nVersículo de ejemplo ({ejemplo['nombre_libro']} "
          f"{ejemplo['numero_capitulo']}:{ejemplo['numero_versiculo']}):")
    print(f"  {ejemplo['texto_original']}")
    print("\nTérminos con mayor peso TF-IDF en ese versículo:")
    for palabra, peso in pesos_ordenados[:5]:
        print(f"  {palabra:15s} {peso:.4f}")


def visualizar_exploratorio(df):
    """Muestra las visualizaciones del análisis exploratorio.

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 4: visualizaciones exploratorias")
    explorador = VisualizacionYExplorador(df)
    explorador.graficoLongitud_Versiculos()
    explorador.graficoVersiculos_por_Libro()

def visualizar_pca(df):
    """Reduce los versículos a dos dimensiones con PCA y los grafica.

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 5: visualización PCA de los versículos")
    tfidf = VectorTF_IDF()
    pca = VisualizacionPCA(df, tfidf, top_n = 300)
    pc1, pc2 = pca.varianza_explicada()
    print(f"Varianza explicada: PC1 {round(pc1,2)}% / PC2 {round(pc2,2)}%")
    pca.graficar("testamento")

def buscar_semantico(df):
    """Demuestra el motor de búsqueda con una frase libre y un versículo real.

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 6: motor de búsqueda semántico")
    buscador = BuscadorSemantico(df, VectorTF_IDF(), Similitud())

    consultas = [
        "love your enemies and pray",
        "In the beginning God created the heavens and the earth",
    ]
    for consulta in consultas:
        print(f"\nConsulta: {consulta!r}")
        resultado = buscador.buscar(consulta, k = 5)
        for _, fila in resultado.iterrows():
            print(f"  [{fila['similitud']:.3f}] {fila['nombre_libro']} "
                  f"{fila['numero_capitulo']}:{fila['numero_versiculo']} - {fila['texto_original']}")

def main():
    df = cargar_preprocesar()
    analizar_vocabulario(df)
    demostrar_tfidf(df)
    visualizar_exploratorio(df)
    visualizar_pca(df)
    buscar_semantico(df)

if __name__ == "__main__":
    main()