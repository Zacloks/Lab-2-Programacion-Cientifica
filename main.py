from src.datos.cargador_datos import CargadorDatos
from src.preprocesamiento.preprocesador_texto import PreprocesadorTexto
from src.preprocesamiento.analizador_vocabulario import AnalizadorVocabulario
from src.representacion.tfidf import VectorTF_IDF
from src.representacion.similitud import SimilitudCoseno
from src.visualizacion.exploratorio import VisualizacionYExplorador
from src.visualizacion.pca import VisualizacionPCA
from src.visualizacion.visualizacion_sentimiento import VisualizacionSentimiento
from src.modelos.buscador_semantico import BuscadorSemantico
from src.modelos.clasificador_versiculos import ClasificadorVersiculos
from src.modelos.generador_texto import ModeloNGramas
from src.modelos.analizador_sentimiento import AnalizadorSentimiento

def cargar_preprocesar():
    """Carga el corpus y le aplica el preprocesamiento (3.1).

    Retorna
    pandas.DataFrame
        El corpus con texto_original, texto y tokens.
    """
    print("\nEtapa 1: carga y preprocesamiento del corpus")
    df = CargadorDatos().cargar_biblia()
    df = PreprocesadorTexto().procesar(df)
    print(f"Versículos cargados: {len(df)}")
    print("\nPrimeras filas del corpus ya preprocesado:")
    print(df[["nombre_libro", "testamento", "texto", "tokens"]].head())
    return df

def analizar_vocabulario(df):
    """Construye el vocabulario y muestra las palabras más frecuentes (3.1).

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 2: vocabulario y frecuencias de palabras")
    analizador = AnalizadorVocabulario()
    analizador.construir_vocabulario(df)
    print(f"Tamaño del vocabulario: {analizador.tamano_vocabulario()} palabras únicas")
    print("\nTop 15 palabras más frecuentes:")
    print(analizador.palabras_mas_comunes(15).to_string(index = False))


def demostrar_tfidf(df):
    """Ajusta el TF-IDF y muestra los términos con más peso de un versículo (3.1).

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
    """Genera las visualizaciones exploratorias, incluido el heatmap (3.2).

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 4: visualizaciones exploratorias")
    explorador = VisualizacionYExplorador(df)
    explorador.graficoLongitud_Versiculos()
    explorador.graficoVersiculos_por_Libro()
    explorador.grafico_palabras_frecuentes()
    explorador.heatmap_similitud_libros()
    print("Gráficos guardados en resultados/")


def visualizar_pca(df):
    """Grafica los versículos en 2D con PCA (3.3).

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 5: visualización PCA de los versículos")
    pca = VisualizacionPCA(df, VectorTF_IDF(), top_n = 300)
    pc1, pc2 = pca.varianza_explicada()
    print(f"Varianza explicada: PC1 {round(pc1, 2)}% / PC2 {round(pc2, 2)}%")
    pca.graficar("testamento")
    print("Gráfico guardado en resultados/")


def buscar_semantico(df):
    """Demuestra la búsqueda con una frase y un versículo real (3.4).

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 6: motor de búsqueda semántico")
    buscador = BuscadorSemantico(df, VectorTF_IDF(), SimilitudCoseno())

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


def clasificar_versiculos(df):
    """Entrena y evalúa el clasificador de libro (3.5).

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 7: clasificador de versículos (predice el libro)")
    clasificador = ClasificadorVersiculos()
    resultado = clasificador.entrenar_evaluar(df)
    print(f"Accuracy en prueba: {resultado['accuracy']:.2%}")
    clasificador.graficar_matriz_confusion(resultado)
    print("Matriz de confusión guardada en resultados/")


def generar_texto(df):
    """Entrena y compara los modelos de n-gramas (3.6).

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 8: generador de versículos falsos (n-gramas)")
    modelos = {
        "Unigram": ModeloNGramas(1),
        "Bigram": ModeloNGramas(2),
        "Trigram": ModeloNGramas(3),
        "4-gram": ModeloNGramas(4),
    }
    for modelo in modelos.values():
        modelo.entrenar(df["tokens"])

    palabra_inicial = "god"
    print(f"\nVersículos generados desde la palabra {palabra_inicial!r}:")
    for nombre, modelo in modelos.items():
        texto = modelo.generar_texto(palabra_inicial = palabra_inicial, longitud_maxima = 15)
        print(f"  {nombre}: {texto}")


def analizar_sentimiento(df):
    """Calcula el sentimiento por versículo y lo agrega por libro (3.7).

    Parámetros
    df : pandas.DataFrame
        Corpus ya preprocesado.
    """
    print("\nEtapa 9: análisis de sentimiento (VADER)")
    analizador = AnalizadorSentimiento()
    df = analizador.analizar(df)

    print("\nDistribución de versículos por sentimiento:")
    print(analizador.distribucion_etiquetas(df).to_string())

    mas_positivos, mas_negativos = analizador.libros_extremos(df, 5)
    print("\nLibros más positivos:")
    print(mas_positivos[["nombre_libro", "sentimiento"]].to_string(index = False))
    print("\nLibros más negativos:")
    print(mas_negativos[["nombre_libro", "sentimiento"]].to_string(index = False))

    visualizacion = VisualizacionSentimiento(df, analizador)
    visualizacion.distribucion_sentimiento()
    visualizacion.evolucion_por_libro()
    visualizacion.evolucion_por_capitulo("Job")
    visualizacion.libros_extremos(10)
    print("Gráficos de sentimiento guardados en resultados/")


def main():
    """Ejecuta todas las etapas del laboratorio en orden."""
    df = cargar_preprocesar()
    analizar_vocabulario(df)
    demostrar_tfidf(df)
    visualizar_exploratorio(df)
    visualizar_pca(df)
    buscar_semantico(df)
    clasificar_versiculos(df)
    generar_texto(df)
    analizar_sentimiento(df)

if __name__ == "__main__":
    main()
