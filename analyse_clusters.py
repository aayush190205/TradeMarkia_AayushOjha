import pickle
import numpy as np
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nLoading clustering artifacts...\n")

# =====================================================
# Load saved artifacts
# =====================================================

with open("data/documents.pkl","rb") as f:
    documents = pickle.load(f)

with open("data/embeddings.pkl","rb") as f:
    embeddings = pickle.load(f)

with open("data/pca_model.pkl","rb") as f:
    pca = pickle.load(f)

with open("data/gmm_model.pkl","rb") as f:
    gmm = pickle.load(f)

# =====================================================
# Compute cluster assignments
# =====================================================

reduced = pca.transform(embeddings)
probs = gmm.predict_proba(reduced)

dominant_clusters = np.argmax(probs, axis=1)
certainty_scores = np.max(probs, axis=1)

# =====================================================
# Compute cluster keywords using TF-IDF
# =====================================================

print("="*70)
print("CLUSTER TOP KEYWORDS")
print("="*70)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    min_df=5
)

X = vectorizer.fit_transform(documents)

feature_names = vectorizer.get_feature_names_out()

cluster_keywords = {}

for cluster_id in np.unique(dominant_clusters):

    indices = np.where(dominant_clusters == cluster_id)[0]

    cluster_matrix = X[indices]

    mean_tfidf = np.asarray(cluster_matrix.mean(axis=0)).flatten()

    top_indices = mean_tfidf.argsort()[-8:][::-1]

    keywords = [feature_names[i] for i in top_indices]

    cluster_keywords[cluster_id] = keywords

    print(f"Cluster {cluster_id} keywords:", ", ".join(keywords))

# =====================================================
# Helper
# =====================================================

def clean_preview(text):
    text = text.replace("\n"," ")
    if "writes:" in text:
        text = text.split("writes:")[-1]
    return textwrap.shorten(text, width=140)

# =====================================================
# CORE CLUSTER MEMBERS
# =====================================================

print("\n")
print("="*70)
print("CORE CLUSTER MEMBERS (HIGH CERTAINTY)")
print("="*70)

most_certain = np.argsort(certainty_scores)[-5:]

for idx in most_certain:

    cluster_probs = probs[idx]
    sorted_clusters = np.argsort(cluster_probs)[::-1]

    c1 = sorted_clusters[0]
    c2 = sorted_clusters[1]

    p1 = cluster_probs[c1]*100
    p2 = cluster_probs[c2]*100

    print("\n--------------------------------------------------")

    print(f"Dominant Cluster : {c1}")
    print(f"Confidence       : {p1:.2f}%")

    print("\nCluster Distribution:")
    print(f"Cluster {c1} → {p1:.2f}%")
    print(f"Cluster {c2} → {p2:.2f}%")

    print("\nCluster Topic Keywords:")
    print("Cluster",c1,":",", ".join(cluster_keywords[c1]))

    print("\nReasoning:")

    print(
        "This document strongly matches the vocabulary used in this cluster. "
        "Because its embedding lies close to the cluster center and far from "
        "other clusters, the Gaussian Mixture Model assigns nearly all "
        "probability mass to this cluster."
    )

    print("\nDocument Preview:")
    print(clean_preview(documents[idx]))

# =====================================================
# BOUNDARY DOCUMENTS
# =====================================================

print("\n")
print("="*70)
print("BOUNDARY DOCUMENTS (MODEL UNCERTAINTY)")
print("="*70)

most_uncertain = np.argsort(certainty_scores)[:5]

for idx in most_uncertain:

    cluster_probs = probs[idx]
    sorted_clusters = np.argsort(cluster_probs)[::-1]

    c1 = sorted_clusters[0]
    c2 = sorted_clusters[1]

    p1 = cluster_probs[c1]*100
    p2 = cluster_probs[c2]*100

    print("\n--------------------------------------------------")

    print("Cluster Distribution:")
    print(f"Cluster {c1} → {p1:.2f}%")
    print(f"Cluster {c2} → {p2:.2f}%")

    print("\nCluster Keywords:")
    print(f"Cluster {c1} :", ", ".join(cluster_keywords[c1]))
    print(f"Cluster {c2} :", ", ".join(cluster_keywords[c2]))

    print("\nReasoning:")

    print(
        "The probabilities of the top two clusters are very close. "
        "This indicates that the document shares vocabulary with both "
        "semantic regions. Because the embedding lies between these "
        "two cluster centers, the Gaussian Mixture Model distributes "
        "probability across both clusters."
    )

    print("\nDocument Preview:")
    print(clean_preview(documents[idx]))