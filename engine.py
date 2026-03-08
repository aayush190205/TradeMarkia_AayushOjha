import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ==============================================================================
# PART 1: EMBEDDING & VECTOR DATABASE SETUP
# ==============================================================================

def clean_document(raw_text):
    # Justification for Data Cleaning: The prompt explicitly asks to discard noise. 
    # The raw UCI 20 Newsgroups files have massive legacy email headers (From:, Subject:, Lines:) 
    # and quoted replies indicated by '>'. 
    # The header usually ends at the first double-newline. I split there to drop it.
    parts = raw_text.split('\n\n', 1)
    if len(parts) > 1:
        body = parts[1]
    else:
        body = raw_text
        
    # Remove quoted lines to ensure the model embeds the user's actual thoughts, 
    # not the historical email chain they are replying to.
    clean_lines = [line for line in body.split('\n') if not line.strip().startswith('>')]
    return '\n'.join(clean_lines).strip()

def load_local_dataset(root_dir):
    print(f"Loading and cleaning ALL files from local directory: {root_dir}")
    documents = []
    labels = []
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Cannot find the folder '{root_dir}'. Please ensure it is in the same directory as this script.")

    categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for category in categories:
        category_path = os.path.join(root_dir, category)
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            
            # Justification for Encoding: These are legacy 1993 text files. Many contain 
            # byte artifacts that break standard utf-8 decoding. Using latin-1 ensures 
            # the ingestion pipeline doesn't crash on these edge cases.
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    raw_text = f.read()
                    
                cleaned_text = clean_document(raw_text)
                
                # Discard the document if it lacks sufficient semantic context after cleaning.
                # Setting a threshold of 100 characters.
                if len(cleaned_text) > 100:
                    documents.append(cleaned_text)
                    labels.append(category)
            except Exception:
                continue 
                
    return documents, labels

def prepare_data_and_embeddings():
    # 1. LOAD AND CLEAN THE FULL DATASET
    documents, labels = load_local_dataset('20_newsgroups')
    print(f"Successfully loaded and cleaned {len(documents)} documents out of the full dataset.")

    # 2. EMBEDDING MODEL SELECTION
    # Justification for Model: I selected 'all-MiniLM-L6-v2'. It maps text to a 384-dimensional 
    # dense vector space. It is a "better fit" than massive LLM embeddings because it runs 
    # efficiently on a CPU, honoring the "lightweight" constraint of the prompt, while still 
    # providing highly accurate cosine-similarity metrics for the semantic cache downstream.
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings for the ENTIRE corpus... (This will take a while on a laptop)")
    embeddings = model.encode(documents, show_progress_bar=True)

    # Normalizing the embeddings upfront allows the cache layer to use a simple, 
    # highly-optimized dot product instead of a full cosine similarity calculation.
    print("Normalizing embeddings...")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # ==============================================================================
    # PART 2: FUZZY CLUSTERING
    # ==============================================================================
    
    print("Applying PCA for dimensionality reduction...")
    # Justification: Density-based and distance-based clustering models degrade in high 
    # dimensional spaces (384 dims). Reducing to 50 dimensions ensures mathematical stability.
    pca = PCA(n_components=50, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Justification for Clustering Approach: The prompt forbids hard cluster assignments 
    # and requires a distribution. Gaussian Mixture Models (GMM) inherently output 
    # probability distributions across all clusters for a given data point.
    # 
    # Justification for K (n_clusters): I selected 15 clusters. While there are 20 source 
    # directories, enforcing 20 clusters is a convenience trap. Semantically, several 
    # categories overlap significantly (e.g., 'comp.sys.mac.hardware' and 'comp.sys.ibm.pc.hardware' 
    # belong to a broader "hardware" semantic space). K=15 provides a more generalized semantic map.
    n_clusters = 15
    print(f"Training Gaussian Mixture Model for {n_clusters} fuzzy clusters...")
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(reduced_embeddings)

    # 3. VECTOR STORE SELECTION
    # Justification for Vector Store: The prompt asks for a vector database but emphasizes a 
    # lightweight build from first principles. For a corpus of this size, standing up a dedicated 
    # service like Milvus or Qdrant introduces unnecessary network overhead. 
    # Persisting serialized NumPy arrays and doing in-memory matrix multiplication is the 
    # fastest, most efficient, and most lightweight "database" possible for this specific constraint.
    print("Persisting vector database and models to disk...")
    os.makedirs("data", exist_ok=True)
    
    with open("data/documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    with open("data/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    with open("data/pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open("data/gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    print("Part 1 & Part 2 Complete! Vector store and clustering models successfully built.")

if __name__ == "__main__":
    prepare_data_and_embeddings()