from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sbn

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from src.representacion.tfidf import VectorTF_IDF

class ClasificadorVersiculos:
    """Predice de qué libro es un versículo (3.5).

    Usa el TF-IDF propio como entrada y un LinearSVC de scikit-learn como
    modelo, que va bien con representaciones dispersas y entrena rápido.
    Acertar el libro exacto entre 66 desde un versículo corto es difícil, así
    que el accuracy es bajo; por eso entrenar_evaluar permite cambiar la
    etiqueta (testamento, género) para comparar granularidades.
    """

    def __init__(self):
        """Inicializa el TF-IDF propio y el modelo LinearSVC."""
        self.tfidf = VectorTF_IDF()
        self.modelo = LinearSVC(max_iter = 5000)
        self.entrenado = False

    def _a_matriz_dispersa(self, serie_vectores):
        """Convierte los vectores TF-IDF dispersos en una matriz dispersa de SciPy.

        Usa el vocabulario del TF-IDF para mapear cada palabra a su columna. Se
        usa matriz dispersa porque la densa tendría casi todo en cero.

        Parámetros
        serie_vectores : Iterable[dict]
            Vectores TF-IDF de los versículos.

        Retorna
        scipy.sparse.csr_matrix
            Matriz de forma (n_versiculos, tamano_vocabulario).
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

        Separa los datos de forma estratificada por la etiqueta, entrena y mide
        el accuracy en prueba.

        Parámetros
        df : pandas.DataFrame
            Corpus con tokens y la columna de etiqueta.
        columna_etiqueta : str
            Clase a predecir (por defecto nombre_libro; también testamento o
            nombre_genero).
        test_size : float
            Proporción para prueba (por defecto 0.2).
        random_state : int
            Semilla para reproducir la partición.

        Retorna
        dict
            Con accuracy, etiquetas, y_test e y_pred.
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
            Tokens del versículo.

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

        Parámetros
        resultado : dict
            Lo que devuelve entrenar_evaluar.
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
        Path("resultados").mkdir(exist_ok = True)
        plt.savefig("resultados/matriz_confusion.png", bbox_inches = "tight")
        plt.close()
