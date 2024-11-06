import argparse  # Permite crear una interfaz de línea de comandos
import json      # Permite manejar archivos JSON para cargar el archivo de lematización
import math      # Proporciona funciones matemáticas como logaritmos y raíces cuadradas
import re        # Librería de expresiones regulares para manipulación de texto
from collections import Counter, defaultdict  # Utilidades para conteo y diccionarios con valores por defecto

import matplotlib.pyplot as plt  # Librería para graficar
import seaborn as sns            # Librería para gráficos estadísticos

# Clase para almacenar información de cada término de un documento
class DocumentTermInfo:
    def __init__(self, term_index, term, tf, idf, tf_idf):
        self.term_index = term_index  # Índice del término
        self.term = term                # Término
        self.tf = tf                    # TF
        self.idf = idf                  # IDF
        self.tf_idf = tf_idf            # TF-IDF

# Clase para almacenar información de cada documento
class DocumentInfo:
    def __init__(self, doc_index):
        self.doc_index = doc_index      # Índice del documento
        self.terms_info = []             # Lista de DocumentTermInfo para este documento

    def add_term_info(self, term_info):
        self.terms_info.append(term_info)  # Añadir información del término a la lista

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
    term_counts = Counter(doc)  # Cuenta las ocurrencias de cada término en el documento
    total_terms = len(doc)       # Total de términos en el documento
    tf = {}
    # Crear un diccionario ordenado por el primer índice de aparición
    for term in doc:
        count = term_counts[term]
        if count > 0:
            # Aplica la fórmula TF = 1 + log10(frecuencia del término)
            tf[term] = 1 + math.log10(count)
    return tf

# Calcula la frecuencia inversa de documento (IDF) para todos los documentos
def calculate_idf(documents):
    idf = {}
    num_docs = len(documents)
    
    # Inicializamos el contador de documentos que contienen cada término
    term_doc_count = defaultdict(int)
    
    # Contamos cuántos documentos contienen cada término
    for doc in documents:
        unique_terms = set(doc)  # Términos únicos en el documento
        for term in unique_terms:
            term_doc_count[term] += 1
            
    # Calculamos IDF usando el inverso logarítmico de la frecuencia del documento
    for term, count in term_doc_count.items():
        if count > 0:  # Aseguramos que no estamos dividiendo por cero
            idf[term] = math.log(num_docs / count)
        else:
            idf[term] = 0  # Valor por defecto si el término no aparece en ningún documento

    return idf

# Calcula TF-IDF para todos los documentos
def calculate_tf_idf(documents):
    tf_idf_docs = []
    idf = calculate_idf(documents)
    for doc in documents:
        tf = calculate_tf(doc)
        tf_idf = {term: tf_val * idf[term] for term, tf_val in tf.items()}
        tf_idf_docs.append(tf_idf)
    return tf_idf_docs, idf

def calculate_vector_length(vector):
    """Calcula la longitud del vector usando la raíz cuadrada de la suma de los cuadrados de los valores."""
    return math.sqrt(sum(value ** 2 for value in vector.values()))

def normalize_vector(vector):
    """Normaliza el vector dividiendo cada valor por la longitud del vector."""
    length = calculate_vector_length(vector)
    if length == 0:
        return vector  # Evita la división por cero
    return {term: value / length for term, value in vector.items()}

def calculate_cosine_similarity(normalized_vec1, normalized_vec2):
    """Calcula la similaridad del coseno entre dos vectores normalizados."""
    common_terms = set(normalized_vec1.keys()).intersection(set(normalized_vec2.keys()))
    return sum(normalized_vec1[term] * normalized_vec2[term] for term in common_terms)

def calculate_cosine_similarities(tf_idf_docs):
    """Calcula la matriz de similaridad del coseno entre todos los documentos."""
    num_docs = len(tf_idf_docs)
    cos_sim_matrix = [[0] * num_docs for _ in range(num_docs)]
    
    # Normalizar los vectores de TF-IDF
    normalized_docs = [normalize_vector(tf_idf) for tf_idf in tf_idf_docs]
    
    for i in range(num_docs):
        for j in range(i, num_docs):
            cos_sim_matrix[i][j] = cos_sim_matrix[j][i] = calculate_cosine_similarity(normalized_docs[i], normalized_docs[j])
    
    return cos_sim_matrix

def write_results_to_file(tf_idf_docs, cos_sim_matrix, output_file, documents_info):
    with open(output_file, 'w') as f:
        for doc_info in documents_info:
            f.write(f"Documento {doc_info.doc_index}:\n")
            f.write(f"{'Index':<10}{'Term':<15}{'TF':<10}{'IDF':<10}{'TF-IDF':<10}\n")
            for term_info in doc_info.terms_info:  # Cambiado a terms_info
                f.write(f"{term_info.term_index:<10}{term_info.term:<15}{term_info.tf:<10.4f}{term_info.idf:<10.4f}{term_info.tf_idf:<10.4f}\n")
            f.write("\n")  # Espacio entre documentos

        # Escribir la matriz de similitudes coseno
        f.write("Similitudes coseno entre documentos:\n")
        for i, row in enumerate(cos_sim_matrix):
            f.write(f"Doc {i + 1}: " + ", ".join(f"{val:<6.4f}" for val in row) + "\n")  # Formatear a 4 decimales

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
    
    # Calcular TF-IDF
    tf_idf_docs, idf = calculate_tf_idf(documents)
    
    # Crear la lista de DocumentInfo para cada documento
    documents_info = []
    for doc_idx, doc in enumerate(documents):
        tf = calculate_tf(doc)  # Calcula TF
        if doc_idx < len(tf_idf_docs):  # Asegúrate de que doc_idx esté dentro de los límites
            for idx, term in enumerate(doc):  # Mantiene el orden de aparición de los términos en el documento
                tf_val = tf.get(term, 0)  # Obtener valor de TF o 0 si no existe
                idf_val = idf.get(term, 0)  # Obtener valor de IDF o 0 si no existe
                
                # Acceder a la tupla de TF-IDF correctamente
                tfidf_val = tf_idf_docs[doc_idx].get(term, 0)  # Obtener valor de TF-IDF o 0 si no existe
                
                # Crear un objeto DocumentTermInfo y añadirlo al DocumentInfo correspondiente
                term_info = DocumentTermInfo(term_index=idx, term=term, tf=tf_val, idf=idf_val, tf_idf=tfidf_val)
                if doc_idx >= len(documents_info):
                    documents_info.append(DocumentInfo(doc_index=doc_idx + 1))  # Doc index es 1-based
                documents_info[doc_idx].add_term_info(term_info)
    
    # Calcular la matriz de similitudes coseno
    cos_sim_matrix = calculate_cosine_similarities(tf_idf_docs)

    # Escribir la información de los documentos en el archivo de salida
    write_results_to_file(tf_idf_docs, cos_sim_matrix, args.output, documents_info)

    # Graficar la matriz de similitud
    plot_similarity_matrix(cos_sim_matrix, args.graph)

if __name__ == "__main__":
    main()