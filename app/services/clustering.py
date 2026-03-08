

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

from app.config import settings


class ClusteringService:


    def __init__(self):
        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None
        self.n_clusters: int = settings.n_clusters
        self.pca_components: int = settings.pca_components
        self.is_fitted: bool = False

    def fit(
        self,
        embeddings: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        
        n_samples, n_features = embeddings.shape
        print(f"[Clustering] Fitting on {n_samples} samples × {n_features} features")

        
        print(f"[Clustering] PCA: {n_features}d → {self.pca_components}d")
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        reduced = self.pca.fit_transform(embeddings)

        variance_explained = self.pca.explained_variance_ratio_.sum()
        print(f"[Clustering] Variance retained: {variance_explained:.3f}")

        
        print(f"[Clustering] Fitting GMM with K={self.n_clusters}, "
              f"covariance={settings.gmm_covariance_type}")

        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=settings.gmm_covariance_type,
            n_init=5,
            max_iter=300,
            random_state=42,
            verbose=1,
        )
        self.gmm.fit(reduced)
        self.is_fitted = True

        
        hard_labels = self.gmm.predict(reduced)
        bic = self.gmm.bic(reduced)
        aic = self.gmm.aic(reduced)

        diagnostics = {
            "n_samples": n_samples,
            "pca_components": self.pca_components,
            "variance_explained": float(variance_explained),
            "n_clusters": self.n_clusters,
            "bic": float(bic),
            "aic": float(aic),
            "converged": self.gmm.converged_,
            "n_iterations": self.gmm.n_iter_,
        }

        
        if n_samples > 10000:
            sample_idx = np.random.RandomState(42).choice(
                n_samples, 10000, replace=False
            )
            sil = silhouette_score(reduced[sample_idx], hard_labels[sample_idx])
        else:
            sil = silhouette_score(reduced, hard_labels)
        diagnostics["silhouette_score"] = float(sil)

        
        if true_labels is not None:
            nmi = normalized_mutual_info_score(true_labels, hard_labels)
            diagnostics["nmi_vs_true_labels"] = float(nmi)

        print(f"[Clustering] BIC={bic:.0f}, AIC={aic:.0f}, "
              f"Silhouette={sil:.3f}")

        return diagnostics

    def evaluate_k_range(
        self,
        embeddings: np.ndarray,
        k_range: range = range(5, 36),
        true_labels: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        
        print(f"[Clustering] Evaluating K in {k_range.start}..{k_range.stop}")

        
        pca = PCA(n_components=self.pca_components, random_state=42)
        reduced = pca.fit_transform(embeddings)
        n_samples = reduced.shape[0]

        
        sil_idx = np.random.RandomState(42).choice(
            n_samples, min(5000, n_samples), replace=False
        )

        results = []
        for k in k_range:
            print(f"  → K={k}...", end=" ", flush=True)
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=settings.gmm_covariance_type,
                n_init=3,
                max_iter=200,
                random_state=42,
            )
            gmm.fit(reduced)

            hard_labels = gmm.predict(reduced)
            bic = gmm.bic(reduced)
            aic = gmm.aic(reduced)
            sil = silhouette_score(reduced[sil_idx], hard_labels[sil_idx])

            result = {
                "k": k,
                "bic": float(bic),
                "aic": float(aic),
                "silhouette_score": float(sil),
                "converged": gmm.converged_,
                "n_iterations": gmm.n_iter_,
                "log_likelihood": float(gmm.score(reduced) * n_samples),
            }

            if true_labels is not None:
                nmi = normalized_mutual_info_score(true_labels, hard_labels)
                result["nmi"] = float(nmi)

            print(f"BIC={bic:.0f}, Sil={sil:.3f}")
            results.append(result)

        return results

    def predict(self, embedding: np.ndarray) -> Dict[str, Any]:
        
        assert self.is_fitted, "Models not fitted. Call fit() or load() first."

        
        emb_2d = embedding.reshape(1, -1)
        reduced = self.pca.transform(emb_2d)

        
        probs = self.gmm.predict_proba(reduced)[0]

        
        dominant = int(np.argmax(probs))

        
        cluster_dist = {
            int(i): round(float(p), 4)
            for i, p in enumerate(probs)
            if p > 0.01
        }

        
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return {
            "dominant_cluster": dominant,
            "cluster_distribution": cluster_dist,
            "entropy": float(entropy),
        }

    def predict_batch(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        assert self.is_fitted, "Models not fitted. Call fit() or load() first."

        reduced = self.pca.transform(embeddings)
        probs = self.gmm.predict_proba(reduced)
        hard_labels = np.argmax(probs, axis=1)

        return hard_labels, probs

    def save(self, directory: Optional[str] = None) -> None:
        
        directory = directory or settings.models_dir
        os.makedirs(directory, exist_ok=True)

        pca_path = os.path.join(directory, "pca.pkl")
        gmm_path = os.path.join(directory, "gmm.pkl")

        with open(pca_path, "wb") as f:
            pickle.dump(self.pca, f)
        with open(gmm_path, "wb") as f:
            pickle.dump(self.gmm, f)

        print(f"[Clustering] Models saved to {directory}")

    def load(self, directory: Optional[str] = None) -> None:
        
        directory = directory or settings.models_dir

        pca_path = os.path.join(directory, "pca.pkl")
        gmm_path = os.path.join(directory, "gmm.pkl")

        if not os.path.exists(pca_path) or not os.path.exists(gmm_path):
            raise FileNotFoundError(
                f"Clustering models not found in {directory}. "
                "Run 'python -m scripts.setup_data' first."
            )

        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        with open(gmm_path, "rb") as f:
            self.gmm = pickle.load(f)

        self.n_clusters = self.gmm.n_components
        self.is_fitted = True
        print(f"[Clustering] Models loaded from {directory} (K={self.n_clusters})")

    def get_boundary_documents(
        self,
        embeddings: np.ndarray,
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        
        assert self.is_fitted, "Models not fitted."

        reduced = self.pca.transform(embeddings)
        probs = self.gmm.predict_proba(reduced)

        
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)


        top_indices = np.argsort(entropy)[-top_n:][::-1]

        boundary_docs = []
        for idx in top_indices:
            top_clusters = np.argsort(probs[idx])[-3:][::-1]
            boundary_docs.append({
                "index": int(idx),
                "entropy": float(entropy[idx]),
                "top_clusters": [
                    {"cluster": int(c), "probability": float(probs[idx, c])}
                    for c in top_clusters
                ],
            })

        return boundary_docs
