
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from app.services.embedder import EmbeddingService
from app.services.clustering import ClusteringService

embedder = EmbeddingService()
embedder.load_model()

clustering = ClusteringService()
clustering.load()

pairs = [
    ("What are the latest developments in space exploration?",
     "Recent news about NASA and space missions"),
    ("Gun control and firearms legislation debate",
     "Should guns be regulated by the government?"),
    ("What is the best graphics card for gaming?",
     "Which GPU should I buy for my computer?"),
    ("Is there a God?",
     "Arguments for and against the existence of God"),
    ("How do I encrypt my email messages?",
     "Email encryption and PGP security"),
]


print("SIMILARITY ANALYSIS FOR CACHE THRESHOLD TUNING")
print("=" * 70)

for q1, q2 in pairs:
    e1 = embedder.embed_query(q1)
    e2 = embedder.embed_query(q2)
    sim = float(e1 @ e2)
    
    c1 = clustering.predict(e1)
    c2 = clustering.predict(e2)
    
    print(f"\nQ1: {q1}")
    print(f"Q2: {q2}")
    print(f"  Cosine similarity: {sim:.4f}")
    print(f"  Q1 cluster: {c1['dominant_cluster']}, Q2 cluster: {c2['dominant_cluster']}")
    print(f"  Same cluster: {c1['dominant_cluster'] == c2['dominant_cluster']}")
    print(f"  Would hit at tau=0.60: {'YES' if sim >= 0.60 else 'NO'}")
    print(f"  Would hit at tau=0.65: {'YES' if sim >= 0.65 else 'NO'}")
    print(f"  Would hit at tau=0.70: {'YES' if sim >= 0.70 else 'NO'}")

print("\n" + "=" * 70)
sims = []
for q1, q2 in pairs:
    e1 = embedder.embed_query(q1)
    e2 = embedder.embed_query(q2)
    sims.append(float(e1 @ e2))


print(f"\nMean similarity of rephrased pairs: {np.mean(sims):.4f}")
print(f"Min similarity: {np.min(sims):.4f}")
print(f"Max similarity: {np.max(sims):.4f}")
print(f"\nRecommendation: Set threshold to ~{np.min(sims) - 0.02:.2f} to catch all rephrasings")
