from cache import SemanticCache

# Thresholds to compare: 
# 0.90 (Too strict), 0.75 (Your choice), 0.50 (Too loose)
thresholds = [0.90, 0.75, 0.50]

# Carefully selected queries to test semantic understanding vs. topic drift
queries = [
    # Group A: Computer Hardware
    "What is the best graphics card for a gaming PC?",
    "Which GPU should I buy for my computer?",     # Semantic Match (Target: Hit at 0.75)
    
    # Group B: Motorcycles
    "I need advice on buying a new Harley Davidson.",
    "What are the best riding helmets for bikes?", # Topic Drift (Target: Miss at 0.75)
]

print("Initializing Semantic Cache for empirical testing...\n")
# Initialize once to load models into memory
test_cache = SemanticCache(threshold=1.0) 

for t in thresholds:
    print(f"\n" + "="*60)
    print(f"EXPERIMENTAL RUN: THRESHOLD = {t}")
    print("="*60)
    
    test_cache.threshold = t
    test_cache.clear_cache()

    for q in queries:
        result = test_cache.process_query(q)
        if result["cache_hit"]:
            print(f"🟢 [CACHE HIT]  Query: '{q}'")
            print(f"    -> Matched with: '{result['matched_query']}'")
            print(f"    -> Similarity Score: {result['similarity_score']:.4f}")
        else:
            print(f"🔴 [CACHE MISS] Query: '{q}' -> Processing & Indexing...")

    stats = test_cache.get_stats()
    print(f"\nFinal Performance Metric: {stats['hit_rate'] * 100:.1f}% Hit Rate")