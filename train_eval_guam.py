import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from typing import Tuple, Dict
from guam_parser import create_guam_train_test_split
from rag_retriever import TimeSeriesEmbedder, UAMTimeSeriesRAG
from radiff_model import RADiffTrajectoryGenerator

def compute_ade_fde(pred_trajs: np.ndarray, target_trajs: np.ndarray) -> Tuple[float, float]:
    """
    Computes Average Displacement Error (ADE) and Final Displacement Error (FDE).
    pred_trajs: (N, pred_len, 3)
    target_trajs: (N, pred_len, 3)
    """
    # Euclidean distance along time step axis
    disp_error = np.linalg.norm(pred_trajs - target_trajs, axis=-1)  # (N, pred_len)
    ade = float(np.mean(disp_error))
    fde = float(np.mean(disp_error[:, -1]))
    return ade, fde

def train_and_evaluate(
    mat_file: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    obs_len: int = 12,
    pred_len: int = 24,
    max_trajectories: int = 500,
    k_neighbors: int = 3,
    save_dir: str = 'saved_models'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Parse and prepare GUAM dataset
    dataset_dict = create_guam_train_test_split(
        mat_file,
        obs_len=obs_len,
        pred_len=pred_len,
        max_trajectories=max_trajectories,
        test_ratio=0.2
    )
    
    train_obs = dataset_dict['train_obs']     # (N_train, obs_len, 3)
    train_pred = dataset_dict['train_pred']   # (N_train, pred_len, 3)
    test_obs = dataset_dict['test_obs']       # (N_test, obs_len, 3)
    test_pred = dataset_dict['test_pred']     # (N_test, pred_len, 3)
    std_scale = dataset_dict['std']           # (3,)
    
    # Step 2: Build Time-Series FAISS RAG index
    print("Building Time-Series RAG vector database...")
    embedder = TimeSeriesEmbedder(target_length=48)
    emb_dim = embedder.embed(train_obs[0]).shape[1]
    
    rag = UAMTimeSeriesRAG(embedding_dim=emb_dim)
    rag.build_index(train_obs, train_pred, embedder)
    
    # Retrieve RAG priors for training and testing datasets
    print(f"Retrieving k={k_neighbors} trajectory priors for train set...")
    train_ret_obs, train_ret_pred = rag.retrieve_nearest(train_obs, embedder, k=k_neighbors)
    print(f"Retrieving k={k_neighbors} trajectory priors for test set...")
    test_ret_obs, test_ret_pred = rag.retrieve_nearest(test_obs, embedder, k=k_neighbors)
    
    # Step 3: Create PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(train_obs, dtype=torch.float32),
        torch.tensor(train_pred, dtype=torch.float32),
        torch.tensor(train_ret_pred, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Step 4: Initialize RADiff Trajectory Diffusion Model
    model = RADiffTrajectoryGenerator(
        obs_len=obs_len,
        pred_len=pred_len,
        context_dim=256,
        num_steps=50
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    print("\n--- Starting RADiff Model Training ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for b_obs, b_pred, b_priors in train_loader:
            b_obs = b_obs.to(device)
            b_pred = b_pred.to(device)
            b_priors = b_priors.to(device)
            
            optimizer.zero_grad()
            loss = model.forward_loss(b_obs, b_pred, b_priors)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(b_obs)
            
        avg_loss = total_loss / len(train_obs)
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Training Diffusion MSE Loss: {avg_loss:.6f}")
            
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds.")
    
    # Step 5: Evaluate Model on Test Dataset
    print("\n--- Evaluating RADiff Model on GUAM Test Set ---")
    model.eval()
    
    test_obs_tensor = torch.tensor(test_obs, dtype=torch.float32).to(device)
    test_priors_tensor = torch.tensor(test_ret_pred, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predicted_pred_norm = model.generate(test_obs_tensor, test_priors_tensor).cpu().numpy()
        
    # Unnormalize to original coordinates (feet)
    predicted_pred_ft = predicted_pred_norm * std_scale + dataset_dict['mean']
    test_pred_ft = test_pred * std_scale + dataset_dict['mean']
    
    # Metrics in normalized space and real-world scale
    ade_norm, fde_norm = compute_ade_fde(predicted_pred_norm, test_pred)
    ade_ft, fde_ft = compute_ade_fde(predicted_pred_ft, test_pred_ft)
    
    print(f"\n================ Evaluation Results ================")
    print(f" Normalized ADE: {ade_norm:.4f} | Normalized FDE: {fde_norm:.4f}")
    print(f" Real Scale ADE: {ade_ft:.2f} ft ({ade_ft*0.3048:.2f} m)")
    print(f" Real Scale FDE: {fde_ft:.2f} ft ({fde_ft*0.3048:.2f} m)")
    print(f"====================================================\n")
    
    # Step 6: Save Model Checkpoint & Evaluation Artifacts
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'radiff_guam_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'dataset_dict': dataset_dict,
        'predicted_test_pred_norm': predicted_pred_norm,
        'test_ret_pred': test_ret_pred,
        'ade_norm': ade_norm,
        'fde_norm': fde_norm,
        'ade_ft': ade_ft,
        'fde_ft': fde_ft
    }, checkpoint_path)
    print(f"Saved model checkpoint to: {checkpoint_path}")
    
    return checkpoint_path, dataset_dict, predicted_pred_norm

if __name__ == '__main__':
    mat_file = r'd:\New folder\GUAM_dir\Generic-Urban-Air-Mobility-GUAM-main\Challenge_Problems\Data_Set_1.mat'
    train_and_evaluate(mat_file, epochs=30, max_trajectories=1500, k_neighbors=5)
