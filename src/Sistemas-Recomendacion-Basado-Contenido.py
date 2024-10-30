import argparse
import json
import math
import re
from collections import Counter, defaultdict

def load_stop_words(stop_file):
    with open(stop_file, 'r') as f:
        stop_words = f.read().splitlines()
    return set(stop_words)

def load_lemmatization(lemmatization_file):
    with open(lemmatization_file, 'r') as f:
        lemmatization_dict = json.load(f)
    return lemmatization_dict

def preprocess_document(doc, stop_words, lemmatization_dict):
    words = re.findall(r'\b\w+\b', doc.lower())
    words = [lemmatization_dict.get(word, word) for word in words if word not in stop_words]
    return words

def load_documents(doc_file, stop_words, lemmatization_dict):
    with open(doc_file, 'r') as f:
        documents = [preprocess_document(line.strip(), stop_words, lemmatization_dict) for line in f]
    return documents

def calculate_tf(doc):
    term_counts = Counter(doc)
    total_terms = len(doc)
    return {term: count / total_terms for term, count in term_counts.items()}

def calculate_idf(documents):
    idf = defaultdict(int)
    num_docs = len(documents)
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            idf[term] += 1
    return {term: math.log(num_docs / (1 + count)) for term, count in idf.items()}

def calculate_tf_idf(documents):
    tf_idf_docs = []
    idf = calculate_idf(documents)
    for doc in documents:
        tf = calculate_tf(doc)
        tf_idf = {term: tf_val * idf[term] for term, tf_val in tf.items()}
        tf_idf_docs.append(tf_idf)
    return tf_idf_docs, idf

def cosine_similarity(doc1, doc2):
    common_terms = set(doc1.keys()).intersection(set(doc2.keys()))
    numerator = sum(doc1[term] * doc2[term] for term in common_terms)
    sum1 = sum(val ** 2 for val in doc1.values())
    sum2 = sum(val ** 2 for val in doc2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return numerator / denominator if denominator != 0 else 0

def calculate_cosine_similarities(tf_idf_docs):
    num_docs = len(tf_idf_docs)
    cos_sim_matrix = [[0] * num_docs for _ in range(num_docs)]
    for i in range(num_docs):
        for j in range(i, num_docs):
            cos_sim_matrix[i][j] = cos_sim_matrix[j][i] = cosine_similarity(tf_idf_docs[i], tf_idf_docs[j])
    return cos_sim_matrix

def write_results_to_file(terms, tf_idf_docs, cos_sim_matrix, output_file):
    with open(output_file, 'w') as f:
        for doc_idx, tf_idf in enumerate(tf_idf_docs):
            f.write(f"\nDocument {doc_idx + 1}:\n")
            f.write(f"{'Index':<10}{'Term':<15}{'TF-IDF':<10}\n")
            for idx, (term, tfidf_val) in enumerate(tf_idf.items()):
                f.write(f"{idx:<10}{term:<15}{tfidf_val:<10.4f}\n")

        f.write("\nCosine Similarities between documents:\n")
        for i, row in enumerate(cos_sim_matrix):
            f.write(f"Doc {i+1}: {row}\n")

def main():
    parser = argparse.ArgumentParser(description="Content-Based Recommender System")
    parser.add_argument("-d", "--documents", required=True, help="Path to the documents file (.txt)")
    parser.add_argument("-s", "--stopwords", required=True, help="Path to the stopwords file (.txt)")
    parser.add_argument("-l", "--lemmatization", required=True, help="Path to the lemmatization file (.json)")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file where results will be saved")
    args = parser.parse_args()

    stop_words = load_stop_words(args.stopwords)
    lemmatization_dict = load_lemmatization(args.lemmatization)
    documents = load_documents(args.documents, stop_words, lemmatization_dict)
    tf_idf_docs, idf = calculate_tf_idf(documents)
    cos_sim_matrix = calculate_cosine_similarities(tf_idf_docs)

    write_results_to_file(list(idf.keys()), tf_idf_docs, cos_sim_matrix, args.output)

if __name__ == "__main__":
    main()