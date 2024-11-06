# Sistema de Recomendación Basado en Contenido

Este proyecto implementa un sistema de recomendación basado en contenido que utiliza las técnicas de TF-IDF y similitud de coseno para recomendar documentos según su contenido textual. Está diseñado para recibir documentos, palabras de parada (stop words) y un diccionario de lematización, generando una matriz de similitudes coseno y un gráfico de calor (heatmap) de las similitudes entre documentos.

## Estructura del Proyecto

- **DocumentTermInfo**: Clase que almacena la información de un término en un documento, incluyendo su índice, TF, IDF y TF-IDF.
- **DocumentInfo**: Clase que almacena la información de todos los términos de un documento, permitiendo un fácil acceso y formato en el archivo de salida.

## Requisitos Previos

Este proyecto requiere Python 3 y las siguientes librerías:

- `argparse` Permite crear una interfaz de línea de comandos
- `json` Permite manejar archivos JSON para cargar el archivo de lematización
- `math` Proporciona funciones matemáticas como logaritmos y raíces cuadradas
- `re` Librería de expresiones regulares para manipulación de texto
- `collections` Utilidades para conteo y diccionarios con valores por defecto
- `matplotlib` Librería para graficar
- `seaborn` Librería para gráficos estadísticos

Puedes instalar las dependencias adicionales ejecutando:

```bash
pip install matplotlib seaborn
```

## Estructura del Código

### Funciones Principales

1. **load_stop_words(stop_file)**: Carga un archivo de palabras de parada y las convierte en un conjunto para eficientar las operaciones de filtrado.

2. **load_lemmatization(lemmatization_file)**: Carga un archivo JSON que contiene un diccionario de lematización y lo convierte en un diccionario de Python.

3. **preprocess_document(doc, stop_words, lemmatization_dict)**: Procesa cada documento convirtiéndolo a minúsculas, eliminando palabras de parada y aplicando lematización según el diccionario dado.

4. **load_documents(doc_file, stop_words, lemmatization_dict)**: Carga todos los documentos de un archivo, aplicando preprocesamiento a cada documento.

5. **calculate_tf(doc)**: Calcula la frecuencia de términos (TF) para un documento usando la fórmula `TF = 1 + log10(frecuencia)`.

6. **calculate_idf(documents)**: Calcula la frecuencia inversa de documentos (IDF) para cada término en el corpus, aplicando la fórmula `IDF = log(num_docs / (1 + num_docs_containing_term))`.

7. **calculate_tf_idf(documents)**: Calcula el valor de TF-IDF para cada término en cada documento y retorna una lista de vectores TF-IDF junto con el IDF de cada término.

8. **calculate_cosine_similarity(vec1, vec2)**: Calcula la similitud del coseno entre dos vectores normalizados.

9. **calculate_cosine_similarities(tf_idf_docs)**: Calcula la matriz de similitud del coseno para todos los documentos en el corpus.

10. **write_results_to_file(tf_idf_docs, cos_sim_matrix, output_file, documents_info)**: Genera un archivo con el TF, IDF y TF-IDF de cada término en cada documento y una tabla de la matriz de similitud del coseno entre documentos.

11. **plot_similarity_matrix(cos_sim_matrix, output_file)**: Crea un gráfico de calor (heatmap) de la matriz de similitud coseno entre documentos y guarda el gráfico en un archivo de salida.

### Clase DocumentTermInfo

La clase `DocumentTermInfo` almacena los valores de TF, IDF y TF-IDF de cada término en un documento junto con su índice. Es utilizada para estructurar la información de términos antes de escribir los resultados en el archivo de salida.

### Clase DocumentInfo

La clase `DocumentInfo` es una estructura que contiene toda la información de un documento específico y permite añadir `DocumentTermInfo` para cada término de dicho documento.

## Ejecución

Este programa se ejecuta desde la línea de comandos. Utiliza los siguientes argumentos para especificar los archivos de entrada y salida:

```bash
python3 src/Sistemas-Recomendacion-Basado-Contenido.py -d <data/examples_documents/path_to_documents.txt> -s <data/stop-words/path_to_stopwords.txt> -l <data/corpus/path_to_corpus.txtn> -o <outputs/path_to_output.txt> -g <outputs/path_to_graph.png>
```

### Argumentos

- `-d` o `--documents`: Ruta al archivo de documentos en formato `.txt`, donde cada línea representa un documento.
- `-s` o `--stopwords`: Ruta al archivo de palabras de parada (stop words) en formato `.txt`.
- `-l` o `--lemmatization`: Ruta al archivo con el diccionario de lematización.
- `-o` o `--output`: Ruta al archivo de salida donde se guardarán los resultados de TF, IDF, TF-IDF y la matriz de similitud coseno.
- `-g` o `--graph`: Ruta del archivo donde se guardará el gráfico de la matriz de similitud coseno.

### Ejemplo de Uso

```bash
python3 src/Sistemas-Recomendacion-Basado-Contenido.py -d data/examples-documents/documents-01.txt -s data/stop-words/stop-words-en.txt -l data/corpus/corpus-en.txt -o outputs/resultado.txt -g outputs/similarity_matrix.png
```

## Resultados

El archivo de salida incluirá:

1. **Para cada documento**:
  - Índice del término
  - Término
  - Valor de TF (Frecuencia de Términos)
  - Valor de IDF (Frecuencia Inversa de Documentos)
  - Valor de TF-IDF
   
2. **Matriz de Similitud Coseno entre Documentos**:
  - Una tabla que muestra los valores de similitud coseno entre cada par de documentos en el corpus.

## Gráfico de la Matriz de Similitud

Además, el archivo de imagen especificado mediante `--graph` generará un gráfico de calor que muestra visualmente la similitud entre los documentos. Este gráfico es útil para identificar rápidamente qué documentos son más similares entre sí.

## Ejemplo de Salida

```plaintext
Documento 1:
Index     Term           TF        IDF       TF-IDF    
0         termino1       0.3010    0.1249    0.0376    
1         termino2       0.4771    0.2218    0.1057    
...

Similitudes coseno entre documentos:
Doc 1: 1.0000, 0.1243, 0.0978, 0.4567, ...
Doc 2: 0.1243, 1.0000, 0.1334, 0.2751, ...
...
```
