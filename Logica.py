# IA orientada a Producto - Análisis de feedback de usuarios de Uber
# MVP: Agrupación y resumen automático de reseñas

# Paso 1: Carga y preprocesamiento del dataset
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar dataset
reviews = pd.read_csv('/Users/cgonzalez/Desktop/DataProductProyect/IADataProduct/csv/uber_reviews_without_reviewid.csv')

# Nos quedamos con columnas clave
df = reviews[['userName', 'content', 'score', 'at', 'appVersion']].dropna(subset=['content'])
# Limpieza simple del texto
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Quitar puntuación
    text = re.sub(r'\s+', ' ', text)  # Quitar espacios extra
    return text.strip().lower()
df['clean_content'] = df['content'].apply(clean_text)
df = df[df['clean_content'].str.split().str.len() > 3]
df = df[~df['clean_content'].duplicated()]

# Paso 2: Embeddings con Sentence-BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['clean_content'].tolist(), show_progress_bar=True)

# Paso 3: Clustering de comentarios
n_clusters = 15  # Se puede ajustar según el dataset
clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = clustering_model.fit_predict(embeddings)
df['cluster'] = cluster_labels

# Paso 4: Resumen automático por clúster (extractivo)
def summarize_cluster(texts, n=3):
    joined_text = ' '.join(texts)
    vectorizer = TfidfVectorizer().fit_transform(texts)
    similarity_matrix = cosine_similarity(vectorizer)
    scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentence for _, sentence in sorted(zip(scores, texts), reverse=True)]
    return ' '.join(ranked_sentences[:n])

cluster_summaries = {}
for label in sorted(df['cluster'].unique()):
    texts = df[df['cluster'] == label]['clean_content'].tolist()
    summary = summarize_cluster(texts)
    cluster_summaries[label] = summary

# Paso 5: Mostrar resultados
for label, summary in cluster_summaries.items():
    print(f"\n\n=== CLUSTER {label} ===")
    print(summary)

# Siguientes pasos:
# - Visualización (ej: wordclouds por cluster)
# - Segmentación por score o versión de app
# - Resúsmenes con LLM si se quiere refinar la calidad
