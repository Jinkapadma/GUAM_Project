import numpy as np
from scipy.signal import resample
from typing import List, Dict, Tuple, Optional

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors

class TimeSeriesEmbedder:
    """Converts 3D time-series observation trajectories into fixed-length embedding vectors."""
    def __init__(self, target_length: int = 64):
        self.target_length = target_length

    def embed(self, time_series: np.ndarray) -> np.ndarray:
        """
        Input: (obs_len, 3) 3D trajectory observation.
        Output: (1, embedding_dim) feature vector.
        """
        resampled = resample(time_series, self.target_length, axis=0)  # (target_length, 3)
        mean_val = np.mean(time_series, axis=0)       # (3,)
        std_val = np.std(time_series, axis=0)         # (3,)
        max_val = np.max(time_series, axis=0)         # (3,)
        min_val = np.min(time_series, axis=0)         # (3,)
        
        features = np.concatenate([resampled.flatten(), mean_val, std_val, max_val, min_val])
        return features.reshape(1, -1).astype(np.float32)

    def embed_batch(self, batch_time_series: np.ndarray) -> np.ndarray:
        """
        Input: (N, obs_len, 3).
        Output: (N, embedding_dim).
        """
        embeddings = [self.embed(ts).squeeze(0) for ts in batch_time_series]
        return np.array(embeddings, dtype=np.float32)

class UAMTimeSeriesRAG:
    """Vector search engine for UAM historical trajectory retrieval (FAISS / sklearn)."""
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.train_obs: Optional[np.ndarray] = None
        self.train_pred: Optional[np.ndarray] = None
        if HAS_FAISS:
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = NearestNeighbors(n_neighbors=5, algorithm='auto')

    def build_index(self, train_obs: np.ndarray, train_pred: np.ndarray, embedder: TimeSeriesEmbedder):
        """Indexes training set observations and associated future targets."""
        self.train_obs = train_obs
        self.train_pred = train_pred
        embeddings = embedder.embed_batch(train_obs)
        if HAS_FAISS:
            self.index.add(embeddings)
            print(f"FAISS index built with {self.index.ntotal} UAM trajectories.")
        else:
            self.index.fit(embeddings)
            print(f"Scikit-Learn NearestNeighbors index built with {len(embeddings)} UAM trajectories.")

    def retrieve_nearest(self, query_obs: np.ndarray, embedder: TimeSeriesEmbedder, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input: query_obs shape (batch_size, obs_len, 3).
        Output:
          retrieved_obs: shape (batch_size, k, obs_len, 3)
          retrieved_pred: shape (batch_size, k, pred_len, 3)
        """
        query_embeddings = embedder.embed_batch(query_obs)
        if HAS_FAISS:
            distances, indices = self.index.search(query_embeddings, k)
        else:
            distances, indices = self.index.kneighbors(query_embeddings, n_neighbors=k)
        
        batch_size = query_obs.shape[0]
        retrieved_obs_list = []
        retrieved_pred_list = []
        
        for b in range(batch_size):
            b_indices = indices[b]
            retrieved_obs_list.append(self.train_obs[b_indices])
            retrieved_pred_list.append(self.train_pred[b_indices])
            
        return np.array(retrieved_obs_list), np.array(retrieved_pred_list)

if __name__ == '__main__':
    embedder = TimeSeriesEmbedder(target_length=32)
    obs = np.random.randn(50, 12, 3)
    pred = np.random.randn(50, 24, 3)
    
    emb_dim = embedder.embed(obs[0]).shape[1]
    rag = UAMTimeSeriesRAG(emb_dim)
    rag.build_index(obs, pred, embedder)
    
    query = np.random.randn(4, 12, 3)
    ret_obs, ret_pred = rag.retrieve_nearest(query, embedder, k=3)
    print("Retrieved obs shape:", ret_obs.shape)
    print("Retrieved pred shape:", ret_pred.shape)
