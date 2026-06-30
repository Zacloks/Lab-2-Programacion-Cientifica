# Laboratorio 2 - Programación Científica

## Integrantes

- Miguel Valenzuela
- Juan Alvarado
- Roger Villarroel

---

## Descripción

Análisis computacional del corpus bíblico usando la versión World English Bible (WEB). El sistema carga y preprocesa el texto, lo representa mediante TF-IDF y aplica varias técnicas encima: visualización, búsqueda por similitud, clasificación de versículos, generación de texto y análisis de sentimiento.

TF-IDF y similitud del coseno están implementados desde cero, sin librerías que los calculen directamente. El resto del pipeline usa pandas, scikit-learn y herramientas estándar de NLP.

---

## Dataset

Se utiliza el dataset **Bible**, disponible en [Kaggle](https://www.kaggle.com/datasets/oswinrh/bible).

Los tres archivos van en la carpeta `data/`:

| Archivo | Contenido |
|---|---|
| `t_web.csv` | Versículos: id, libro, capítulo, versículo, texto |
| `key_english.csv` | Nombre y testamento por libro |
| `key_genre_english.csv` | Género literario por libro |

Cada versículo queda asociado a su libro, capítulo, testamento y género tras la carga.

---

## Estructura

```text
Lab-2-Programacion-Cientifica/
├── data/
├── resultados/
├── src/
│   ├── datos/
│   │   └── cargador_datos.py
│   ├── preprocesamiento/
│   │   ├── preprocesador_texto.py
│   │   └── analizador_vocabulario.py
│   ├── representacion/
│   │   ├── tfidf.py
│   │   └── similitud.py
│   ├── visualizacion/
│   │   ├── exploratorio.py
│   │   ├── pca.py
│   │   └── visualizacion_sentimiento.py
│   └── modelos/
│       ├── buscador_semantico.py
│       ├── clasificador_versiculos.py
│       ├── generador_texto.py
│       └── analizador_sentimiento.py
└── main.py
```

| Módulo | Responsabilidad |
|---|---|
| `datos` | Carga y unión de los CSV |
| `preprocesamiento` | Limpieza, tokenización y vocabulario |
| `representacion` | TF-IDF y similitud del coseno |
| `visualizacion` | Gráficos exploratorios, PCA y sentimiento |
| `modelos` | Buscador, clasificador, generador y análisis de sentimiento |

---

## Requisitos

- Python 3.11
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk

---

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Zacloks/Lab-2-Programacion-Cientifica.git
cd Lab-2-Programacion-Cientifica
```

### 2. Instalar dependencias
**Con conda:**
```bash
conda env create -f environment.yml
conda activate taller1
```
**Con pip:**
```bash
pip install -r requirements.txt
```
### 3. Ejecutar
```bash
python main.py
```