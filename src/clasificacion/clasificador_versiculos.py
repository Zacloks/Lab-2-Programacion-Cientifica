import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

from src.representacion.tfidf import VectorTF_IDF

class ClasificadorVersiculos:
    """Clasificador que predice a qué libro pertenece un versículo.

    Implementa el requerimiento 3.5: dado un versículo, predice su libro de
    origen. La entrada es la representación TF-IDF del versículo (calculada con
    la implementación propia del equipo) y la etiqueta es el nombre del libro.

    Se usa una máquina de vectores de soporte lineal (LinearSVC) como
    clasificador. Es uno de los modelos más sólidos para clasificación de texto
    con representaciones dispersas y de alta dimensión como TF-IDF, y entrena
    rápido sobre las decenas de miles de versículos del corpus. La pauta solo
    exige implementar a mano TF-IDF y la similitud del coseno, por lo que aquí
    sí se puede apoyar en scikit-learn para el modelo y las métricas.

    Predecir el libro exacto entre 66 posibles a partir de un único versículo
    corto es una tarea muy difícil: los versículos son breves y comparten un
    vocabulario común (Dios, rey, dijo...), así que el accuracy a nivel de libro
    es bajo y apenas supera la baseline de predecir siempre el libro más grande.
    Por eso el clasificador es parametrizable por etiqueta (entrenar_evaluar
    acepta columna_etiqueta): con "nombre_libro" resuelve el requisito 3.5,
    mientras que con "testamento" o "nombre_genero" se observa que el modelo sí
    captura señal a granularidad más gruesa, un contraste muy útil para el
    análisis del informe.
    """

    def __init__(self):
        """Inicializa el vectorizador TF-IDF propio y el modelo LinearSVC."""
        self.tfidf = VectorTF_IDF()
        self.modelo = LinearSVC(max_iter = 5000)
        self.entrenado = False

    def _a_matriz_dispersa(self, serie_vectores):
        """Convierte los vectores TF-IDF dispersos en una matriz dispersa SciPy.

        Cada vector es un diccionario palabra a peso; se traduce cada palabra a
        su índice de columna usando el vocabulario aprendido por el TF-IDF y se
        arma una matriz dispersa en formato CSR, apta para scikit-learn. Se usa
        una matriz dispersa porque la densa (versículos x vocabulario) tendría
        cientos de millones de celdas casi todas en cero.

        Parámetros
        serie_vectores : Iterable[dict]
            Vectores TF-IDF de los versículos.

        Retorna
        scipy.sparse.csr_matrix
            Matriz de características de forma (n_versiculos, tamano_vocabulario).
        """
        indices_columna = []
        indices_fila = [0]
        datos = []

        for vector in serie_vectores:
            for termino, peso in vector.items():
                columna = self.tfidf.vocabulario.get(termino)
                if columna is not None:
                    indices_columna.append(columna)
                    datos.append(peso)
            indices_fila.append(len(indices_columna))

        n_filas = len(indices_fila) - 1
        n_columnas = len(self.tfidf.vocabulario)
        return csr_matrix((datos, indices_columna, indices_fila),
                          shape = (n_filas, n_columnas))

    def entrenar_evaluar(self, df, columna_etiqueta = "nombre_libro",
                         test_size = 0.2, random_state = 42):
        """Entrena el clasificador y lo evalúa sobre un conjunto de prueba.

        Separa los versículos en entrenamiento y prueba de forma estratificada
        por la etiqueta (para que todas las clases estén representadas en ambos
        conjuntos), entrena el modelo con la partición de entrenamiento y mide
        el accuracy sobre la de prueba.

        Parámetros
        df : pandas.DataFrame
            Corpus con la columna tokens y la columna de etiqueta elegida.
        columna_etiqueta : str
            Columna que se usa como clase a predecir. Por defecto nombre_libro,
            que resuelve el requisito 3.5; también admite, por ejemplo,
            testamento o nombre_genero para analizar granularidades más gruesas.
        test_size : float
            Proporción de versículos reservados para prueba (por defecto 0.2).
        random_state : int
            Semilla para que la partición sea reproducible.

        Retorna
        dict
            Con las claves accuracy, etiquetas (las clases en orden),
            y_test e y_pred, para poder calcular la matriz de confusión.
        """
        self.tfidf.ajustar(df["tokens"])
        vectores = self.tfidf.transformar(df["tokens"])
        X = self._a_matriz_dispersa(vectores)
        y = df[columna_etiqueta].to_numpy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = random_state, stratify = y
        )

        self.modelo.fit(X_train, y_train)
        self.entrenado = True
        y_pred = self.modelo.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "etiquetas": self.modelo.classes_,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    def predecir(self, tokens):
        """Predice el libro de un versículo ya tokenizado.

        Parámetros
        tokens : list[str]
            Tokens del versículo (limpios y sin stopwords).

        Retorna
        str
            Nombre del libro predicho.
        """
        if not self.entrenado:
            raise RuntimeError("El clasificador debe entrenarse antes de predecir.")
        vector = self.tfidf.calcularTF_IDF_Documento(tokens)
        X = self._a_matriz_dispersa([vector])
        return self.modelo.predict(X)[0]

    def graficar_matriz_confusion(self, resultado):
        """Grafica la matriz de confusión del conjunto de prueba.

        Cada celda (i, j) cuenta cuántos versículos del libro i se predijeron
        como del libro j. Una diagonal marcada indica buenas predicciones; los
        valores fuera de la diagonal revelan entre qué libros se confunde el
        modelo (típicamente libros con vocabulario o temática parecida).

        Parámetros
        resultado : dict
            El diccionario devuelto por entrenar_evaluar.
        """
        etiquetas = resultado["etiquetas"]
        matriz = confusion_matrix(resultado["y_test"], resultado["y_pred"],
                                  labels = etiquetas)

        plt.figure(figsize = (18, 15))
        sbn.heatmap(matriz, cmap = "viridis", square = True,
                    xticklabels = etiquetas, yticklabels = etiquetas,
                    cbar_kws = {"label": "Cantidad de versículos"})
        plt.title(f"Matriz de confusión del clasificador "
                  f"(accuracy = {resultado['accuracy']:.2%})")
        plt.xlabel("Libro predicho")
        plt.ylabel("Libro real")
        plt.tight_layout()
        plt.show()
