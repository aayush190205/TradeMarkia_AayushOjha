import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, threshold=0.75):
        # =========================================================================
        # THE TUNABLE DECISION: COSINE SIMILARITY THRESHOLD
        # =========================================================================
        # Justification: The threshold of 0.75 is selected as the 'Goldilocks' value. 
        # It is high enough to prevent 'Topic Drift' (matching unrelated queries) 
        # but low enough to account for natural language variations, such as 
        # synonyms (e.g., 'GPU' vs 'Graphics Card') and different sentence structures.
        # - 0.90+ (Too Strict): Fails to capture semantic synonyms like 'GPU' vs 'Graphics Card'.
        # - 0.60 (Too Loose): Risks 'Topic Drift' where unrelated queries match due to shared keywords.
        # - 0.75 (Optimal): Provides the best balance, significantly increasing cache hit rates 
        #   for natural language variations while maintaining high precision.
        self.threshold = threshold
        
        
        # =========================================================================
        # CACHE DATA STRUCTURE & CLUSTER INTEGRATION
        # Requirement: "The cluster structure... should be doing real work here."
        # =========================================================================
        # Justification: I am not using Redis or any external libraries. 
        # The cache structure is a dictionary partitioned by cluster ID: 
        # { cluster_id: [ {"query": "...", "vector": np.array, "result": "..."} ] }
        # If the cache grows to millions of entries, an O(N) linear scan becomes a bottleneck.
        # By partitioning the cache by the dominant cluster ID, I reduce the search space 
        # from O(N) to roughly O(N/K). We only compute similarity against cached queries 
        # that live in the same semantic neighborhood.
        self.store = {}
        
        self.stats = {
            "total_entries": 0,
            "hit_count": 0,
            "miss_count": 0
        }

        print("Initializing Semantic Cache: Loading models and vector database into memory...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load the artifacts we generated in Part 1 & 2
        with open("data/documents.pkl", "rb") as f:
            self.corpus_docs = pickle.load(f)
        with open("data/embeddings.pkl", "rb") as f:
            self.corpus_embs = pickle.load(f)
        with open("data/pca_model.pkl", "rb") as f:
            self.pca = pickle.load(f)
        with open("data/gmm_model.pkl", "rb") as f:
            self.gmm = pickle.load(f)
            
        print("Cache engine initialized successfully.")

    def _get_dominant_cluster(self, embedding):
        """Reduces dimensionality and asks the GMM for the highest probability cluster."""
        reduced = self.pca.transform(embedding)
        probabilities = self.gmm.predict_proba(reduced)[0]
        # The prompt asks for a distribution, but for the specific job of routing 
        # an incoming query to the correct cache partition, we route based on the 
        # *dominant* cluster (highest probability).
        return int(np.argmax(probabilities))
    

    # =========================================================================
    # VECTOR SEARCH FUNCTION (CORPUS SEARCH)
    # =========================================================================
    # Justification:
    # Instead of returning only one document, this function returns the Top-K
    # most semantically similar documents. Real semantic search systems typically
    # return multiple relevant results because several documents may match the
    # intent of the query.
    #
    # Returning Top-K improves robustness and allows the API to behave more like
    # a real semantic search engine.
    #
    def _search_corpus(self, query_vector, k=3):
        """Fallback vector search across the entire dataset when there is a cache miss."""
        
        # Because the corpus embeddings were normalized during ingestion, we can use 
        # a highly optimized dot product here instead of computing full cosine similarity.
        similarities = np.dot(self.corpus_embs, query_vector.T).flatten()

        # Identify the indices of the top-k most similar documents
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        results = []

        for idx in top_k_indices:
            results.append({
                "document": self.corpus_docs[idx],
                "score": float(similarities[idx])
            })

        return results


    def process_query(self, user_query):
        # =========================================================================
        # STEP 1: Embed the incoming query and normalize it
        # =========================================================================
        raw_emb = self.embed_model.encode([user_query])
        query_vector = raw_emb / np.linalg.norm(raw_emb, axis=1, keepdims=True)
        
        # =========================================================================
        # STEP 2: Find which cluster bucket this query belongs to
        # =========================================================================
        dominant_cluster = self._get_dominant_cluster(raw_emb)
        
        # =========================================================================
        # STEP 3: Initialize the cluster bucket in our cache if it doesn't exist yet
        # =========================================================================
        if dominant_cluster not in self.store:
            self.store[dominant_cluster] = []

        # =========================================================================
        # STEP 4: Check the cache (ONLY within the relevant cluster to save compute)
        # =========================================================================
        bucket = self.store[dominant_cluster]
        best_sim = -1.0
        best_match = None
        
        if len(bucket) > 0:
            # Stack all cached vectors in this specific bucket and perform
            # a vectorized similarity computation
            cached_vectors = np.vstack([item["vector"] for item in bucket])
            similarities = np.dot(cached_vectors, query_vector.T).flatten()
            
            max_idx = np.argmax(similarities)
            best_sim = float(similarities[max_idx])
            
            if best_sim >= self.threshold:
                # CACHE HIT
                self.stats["hit_count"] += 1
                best_match = bucket[max_idx]

                return {
                    "query": user_query,
                    "cache_hit": True,
                    "matched_query": best_match["query"],
                    "similarity_score": round(best_sim, 4),
                    "result": best_match["result"],
                    "dominant_cluster": dominant_cluster
                }

        # =========================================================================
        # STEP 5: CACHE MISS → PERFORM VECTOR SEARCH ON CORPUS
        # =========================================================================
        self.stats["miss_count"] += 1

        result = self._search_corpus(query_vector, k=3)
        
        # =========================================================================
        # STEP 6: STORE RESULT IN CACHE FOR FUTURE REUSE
        # =========================================================================
        self.store[dominant_cluster].append({
            "query": user_query,
            "vector": query_vector[0],  # Store the 1D vector
            "result": result
        })

        self.stats["total_entries"] += 1

        return {
            "query": user_query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result,
            "dominant_cluster": dominant_cluster
        }


    def get_stats(self):
        total_requests = self.stats["hit_count"] + self.stats["miss_count"]
        hit_rate = (self.stats["hit_count"] / total_requests) if total_requests > 0 else 0.0

        return {
            "total_entries": self.stats["total_entries"],
            "hit_count": self.stats["hit_count"],
            "miss_count": self.stats["miss_count"],
            "hit_rate": round(hit_rate, 4)
        }


    def clear_cache(self):
        self.store = {}

        self.stats = {
            "total_entries": 0,
            "hit_count": 0,
            "miss_count": 0
        }