import json
import os
import sys
import time

import numpy as np
from sklearn.datasets import fetch_20newsgroups

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.services.clustering import ClusteringService
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.utils.preprocessing import clean_corpus


def main():
    start_time = time.time()

    print("=" * 70)
    print("SEMANTIC SEARCH SYSTEM — DATA SETUP PIPELINE")
    print("=" * 70)

    os.makedirs(settings.models_dir, exist_ok=True)
    os.makedirs(settings.plots_dir, exist_ok=True)

    print("\n" + "─" * 70)
    print("STEP 1: Loading 20 Newsgroups dataset")
    print("─" * 70)

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    print(f"  Loaded {len(dataset.data)} documents across {len(dataset.target_names)} categories")
    print(f"  Categories: {dataset.target_names}")

    print("\n" + "─" * 70)
    print("STEP 2: Cleaning corpus")
    print("─" * 70)

    cleaned_docs, valid_indices, valid_categories = clean_corpus(
        documents=dataset.data,
        categories=dataset.target,
        category_names=dataset.target_names,
        min_length=30,
    )

    category_names = [dataset.target_names[c] for c in valid_categories]

    print("\n" + "─" * 70)
    print("STEP 3: Initializing services")
    print("─" * 70)

    embedder = EmbeddingService()
    embedder.load_model()

    vector_store = VectorStoreService()
    vector_store.initialize()

    print("\n" + "─" * 70)
    print("STEP 4: Embedding documents & storing in ChromaDB")
    print("─" * 70)

    if vector_store.document_count >= len(cleaned_docs):
        print(f"  ✓ ChromaDB already has {vector_store.document_count} documents. Skipping.")
        embeddings, doc_ids, _ = vector_store.get_all_embeddings()
    else:
        embeddings = embedder.embed_documents(cleaned_docs, batch_size=256)

        metadatas = [
            {
                "original_index": int(valid_indices[i]),
                "category": category_names[i],
                "category_id": int(valid_categories[i]),
            }
            for i in range(len(cleaned_docs))
        ]

        vector_store.add_documents(
            documents=cleaned_docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(f"  Total documents in store: {vector_store.document_count}")

    print("\n" + "─" * 70)
    print("STEP 5: Evaluating cluster counts (K=5..35)")
    print("─" * 70)

    clustering = ClusteringService()
    true_labels = np.array(valid_categories)

    k_results = clustering.evaluate_k_range(
        embeddings=embeddings,
        k_range=range(5, 36),
        true_labels=true_labels,
    )

    k_results_path = os.path.join(settings.models_dir, "k_evaluation.json")
    with open(k_results_path, "w") as f:
        json.dump(k_results, f, indent=2)
    print(f"  Saved K evaluation to {k_results_path}")

    best_by_bic = min(k_results, key=lambda x: x["bic"])
    best_by_sil = max(k_results, key=lambda x: x["silhouette_score"])
    print(f"\n  Best K by BIC:        {best_by_bic['k']} (BIC={best_by_bic['bic']:.0f})")
    print(f"  Best K by Silhouette: {best_by_sil['k']} (Sil={best_by_sil['silhouette_score']:.3f})")

    if "nmi" in k_results[0]:
        best_by_nmi = max(k_results, key=lambda x: x["nmi"])
        print(f"  Best K by NMI:        {best_by_nmi['k']} (NMI={best_by_nmi['nmi']:.3f})")

    print("\n" + "─" * 70)
    print(f"STEP 6: Fitting final GMM with K={settings.n_clusters}")
    print("─" * 70)

    clustering.n_clusters = settings.n_clusters
    diagnostics = clustering.fit(embeddings=embeddings, true_labels=true_labels)

    diag_path = os.path.join(settings.models_dir, "clustering_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  Saved diagnostics to {diag_path}")

    clustering.save()

    print("\n" + "─" * 70)
    print("STEP 7: Assigning clusters to documents in ChromaDB")
    print("─" * 70)

    hard_labels, probs = clustering.predict_batch(embeddings)

    _, doc_ids, existing_metadatas = vector_store.get_all_embeddings()

    updated_metadatas = []
    for i, meta in enumerate(existing_metadatas):
        top_clusters = np.argsort(probs[i])[-5:][::-1]
        cluster_dist = {
            int(c): round(float(probs[i, c]), 4) for c in top_clusters
        }

        updated_meta = dict(meta)
        updated_meta["dominant_cluster"] = int(hard_labels[i])
        updated_meta["cluster_distribution"] = json.dumps(cluster_dist)
        updated_metadatas.append(updated_meta)

    vector_store.update_metadata(doc_ids, updated_metadatas)

    print("\n" + "─" * 70)
    print("STEP 8: Boundary document analysis")
    print("─" * 70)

    boundary_docs = clustering.get_boundary_documents(embeddings, top_n=20)

    print(f"\n  Top 10 most ambiguous documents (highest entropy):")
    print(f"  {'Index':>7} {'Entropy':>8}  Top Clusters")
    print(f"  {'─'*7} {'─'*8}  {'─'*40}")

    for doc in boundary_docs[:10]:
        idx = doc["index"]
        category = category_names[idx] if idx < len(category_names) else "?"
        clusters_str = ", ".join(
            f"C{c['cluster']}:{c['probability']:.2f}"
            for c in doc["top_clusters"]
        )
        print(f"  {idx:>7} {doc['entropy']:>8.3f}  [{category}] {clusters_str}")

    boundary_path = os.path.join(settings.models_dir, "boundary_documents.json")
    for doc in boundary_docs:
        idx = doc["index"]
        if idx < len(cleaned_docs):
            doc["text_preview"] = cleaned_docs[idx][:200]
            doc["category"] = category_names[idx]
    with open(boundary_path, "w") as f:
        json.dump(boundary_docs, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"  Total time:     {elapsed:.1f}s")
    print(f"  Documents:      {vector_store.document_count}")
    print(f"  Dimensions:     {embedder.dimension}")
    print(f"  PCA components: {settings.pca_components}")
    print(f"  Clusters:       {settings.n_clusters}")
    print(f"  Variance kept:  {diagnostics['variance_explained']:.1%}")
    print(f"  Silhouette:     {diagnostics['silhouette_score']:.3f}")
    if "nmi_vs_true_labels" in diagnostics:
        print(f"  NMI vs labels:  {diagnostics['nmi_vs_true_labels']:.3f}")
    print(f"\n  Start the API server with:")
    print(f"    uvicorn app.main:app --host 0.0.0.0 --port 8000")
    print("=" * 70)


if __name__ == "__main__":
    main()
