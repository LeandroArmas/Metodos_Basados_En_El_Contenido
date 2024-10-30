import argparse  # Permite crear una interfaz de línea de comandos
import json      # Permite manejar archivos JSON para cargar el archivo de lematización
import math      # Proporciona funciones matemáticas como logaritmos y raíces cuadradas
import re        # Librería de expresiones regulares para manipulación de texto
from collections import Counter, defaultdict  # Utilidades para conteo y diccionarios con valores por defecto

import matplotlib.pyplot as plt
import seaborn as sns

# Carga el archivo de palabras de parada y las convierte en un conjunto
def load_stop_words(stop_file):
    with open(stop_file, 'r') as f:
        stop_words = f.read().splitlines()  # Lee el archivo línea por línea y elimina saltos de línea
    return set(stop_words)  # Convierte la lista de palabras de parada en un conjunto para eficiencia

# Carga el archivo de lematización en formato JSON
def load_lemmatization(lemmatization_file):
    with open(lemmatization_file, 'r') as f:
        lemmatization_dict = json.load(f)  # Lee y convierte el archivo JSON a un diccionario de lematización
    return lemmatization_dict

# Procesa un documento: convierte a minúsculas, elimina palabras de parada, y aplica lematización
def preprocess_document(doc, stop_words, lemmatization_dict):
    # Extrae palabras usando una expresión regular que busca palabras alfanuméricas
    words = re.findall(r'\b\w+\b', doc.lower())
    # Aplica lematización y elimina palabras de parada
    words = [lemmatization_dict.get(word, word) for word in words if word not in stop_words]
    return words

# Carga y preprocesa todos los documentos de un archivo
def load_documents(doc_file, stop_words, lemmatization_dict):
    with open(doc_file, 'r') as f:
        # Procesa cada línea del archivo como un documento
        documents = [preprocess_document(line.strip(), stop_words, lemmatization_dict) for line in f]
    return documents

# Calcula la frecuencia de término (TF) para un documento
def calculate_tf(doc):
    term_counts = Counter(doc)  # Cuenta ocurrencias de cada término en el documento
    total_terms = len(doc)  # Calcula el número total de términos en el documento
    # Devuelve la frecuencia de cada término dividiendo por el total de términos
    return {term: count / total_terms for term, count in term_counts.items()}

# Calcula la frecuencia inversa de documento (IDF) para todos los documentos
def calculate_idf(documents):
    idf = defaultdict(int)  # Diccionario para contar en cuántos documentos aparece cada término
    num_docs = len(documents)  # Número total de documentos
    for doc in documents:
        unique_terms = set(doc)  # Obtiene términos únicos en el documento
        for term in unique_terms:
            idf[term] += 1  # Incrementa el conteo de documentos para cada término
    # Calcula IDF para cada término usando el logaritmo de (num_docs / (1 + count))
    return {term: math.log(num_docs / (1 + count)) for term, count in idf.items()}

# Calcula TF-IDF para todos los documentos
def calculate_tf_idf(documents):
    tf_idf_docs = []
    idf = calculate_idf(documents)  # Calcula IDF de todos los documentos
    for doc in documents:
        tf = calculate_tf(doc)  # Calcula TF para el documento actual
        # Calcula TF-IDF multiplicando TF por IDF para cada término
        tf_idf = {term: tf_val * idf[term] for term, tf_val in tf.items()}
        tf_idf_docs.append(tf_idf)  # Añade el resultado TF-IDF del documento a la lista
    return tf_idf_docs, idf

# Calcula la similitud coseno entre dos documentos
def cosine_similarity(doc1, doc2):
    # Encuentra términos comunes en ambos documentos
    common_terms = set(doc1.keys()).intersection(set(doc2.keys()))
    # Calcula el numerador sumando el producto de TF-IDF para los términos comunes
    numerator = sum(doc1[term] * doc2[term] for term in common_terms)
    # Calcula el denominador usando las magnitudes de los vectores TF-IDF
    sum1 = sum(val ** 2 for val in doc1.values())
    sum2 = sum(val ** 2 for val in doc2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    # Devuelve la similitud coseno o 0 si el denominador es 0
    return numerator / denominator if denominator != 0 else 0

# Calcula la matriz de similitud coseno para todos los documentos
def calculate_cosine_similarities(tf_idf_docs):
    num_docs = len(tf_idf_docs)
    # Inicializa una matriz de ceros para almacenar similitudes
    cos_sim_matrix = [[0] * num_docs for _ in range(num_docs)]
    for i in range(num_docs):
        for j in range(i, num_docs):  # Solo calcula la mitad superior de la matriz (simétrica)
            cos_sim_matrix[i][j] = cos_sim_matrix[j][i] = cosine_similarity(tf_idf_docs[i], tf_idf_docs[j])
    return cos_sim_matrix

def write_results_to_file(documents, tf_idf_docs, cos_sim_matrix, idf, output_file):
    with open(output_file, 'w') as f:
        # Para cada documento, escribe los valores de TF, IDF y TF-IDF para cada término
        for doc_idx, tf_idf in enumerate(tf_idf_docs):
            f.write(f"\nDocument {doc_idx + 1}:\n")
            f.write(f"{'Index':<10}{'Term':<15}{'TF':<10}{'IDF':<10}{'TF-IDF':<10}\n")  # Cabecera de columnas
            
            # Calcular TF para el documento actual
            tf = calculate_tf(documents[doc_idx])
            
            # Escribir cada término con su índice, TF, IDF y valor TF-IDF
            for idx, term in enumerate(tf_idf.keys()):
                tf_val = tf[term]            # Valor TF del término en el documento
                idf_val = idf[term]          # Valor IDF del término en el corpus
                tfidf_val = tf_idf[term]     # Valor TF-IDF del término en el documento
                f.write(f"{idx:<10}{term:<15}{tf_val:<10.4f}{idf_val:<10.4f}{tfidf_val:<10.4f}\n")
        
        # Escribir la matriz de similitud coseno entre documentos
        f.write("\nCosine Similarities between documents:\n")
        for i, row in enumerate(cos_sim_matrix):
            f.write(f"Doc {i+1}: {row}\n")

def plot_similarity_matrix(cos_sim_matrix, output_file):
    # Crear un gráfico de calor (heatmap) para la matriz de similaridad
    plt.figure(figsize=(10, 8))
    
    # Configurar el gráfico de calor
    sns.heatmap(cos_sim_matrix, annot=True, fmt=".1f", cmap='coolwarm', square=True,
                cbar_kws={"shrink": .8}, 
                xticklabels=[f"Doc {i+1}" for i in range(len(cos_sim_matrix))],
                yticklabels=[f"Doc {i+1}" for i in range(len(cos_sim_matrix))])
    
    plt.title('Cosine Similarity Matrix')
    plt.xlabel('Documents')
    plt.ylabel('Documents')
    plt.savefig(output_file)
    plt.show()


# Función principal para la interfaz de línea de comandos
def main():
    # Define los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Content-Based Recommender System")
    parser.add_argument("-d", "--documents", required=True, help="Path to the documents file (.txt)")
    parser.add_argument("-s", "--stopwords", required=True, help="Path to the stopwords file (.txt)")
    parser.add_argument("-l", "--lemmatization", required=True, help="Path to the lemmatization file (.json)")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file where results will be saved")
    parser.add_argument("-g", "--graph", required=True, help="Path to the output file for the similarity matrix graph")
    args = parser.parse_args()

    # Carga archivos y calcula TF-IDF y similitudes
    stop_words = load_stop_words(args.stopwords)
    lemmatization_dict = load_lemmatization(args.lemmatization)
    documents = load_documents(args.documents, stop_words, lemmatization_dict)
    tf_idf_docs, idf = calculate_tf_idf(documents)
    cos_sim_matrix = calculate_cosine_similarities(tf_idf_docs)

    # Escribe resultados en el archivo de salida
    write_results_to_file(documents, tf_idf_docs, cos_sim_matrix, idf, args.output)
    
    # Llamar a la función para crear el gráfico
    plot_similarity_matrix(cos_sim_matrix, args.graph)

# Ejecuta el programa
if __name__ == "__main__":
    main()