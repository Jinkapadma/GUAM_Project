import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_uam_predictions(checkpoint_path: str = 'saved_models/radiff_guam_model.pt', output_path: str = 'uam_trajectory_prediction.png', sample_indices: list = [0, 1, 2]):
    """
    Plots Ground Truth vs RADiff Predicted 3D UAM Flight Trajectories and saves PNG visualization.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found!")
        return
        
    print(f"Loading checkpoint for visualization from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    dataset_dict = checkpoint['dataset_dict']
    pred_test_norm = checkpoint['predicted_test_pred_norm']  # (N_test, pred_len, 3)
    
    test_obs_norm = dataset_dict['test_obs']                  # (N_test, obs_len, 3)
    test_pred_norm = dataset_dict['test_pred']                # (N_test, pred_len, 3)
    
    mean = dataset_dict['mean']                                # (3,)
    std = dataset_dict['std']                                  # (3,)
    
    # Create matplotlib figure with subplots
    fig = plt.figure(figsize=(18, 6), dpi=120)
    
    for i, idx in enumerate(sample_indices):
        if idx >= len(test_obs_norm):
            continue
            
        # Convert normalized values back to original coordinates (feet)
        obs_ft = test_obs_norm[idx] * std + mean
        gt_pred_ft = test_pred_norm[idx] * std + mean
        pred_ft = pred_test_norm[idx] * std + mean
        
        # 3D Subplot
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        # Plot observed history
        ax.plot(obs_ft[:, 0], obs_ft[:, 1], -obs_ft[:, 2], label='Observed History (Past)', color='#0072BD', linewidth=2.5, marker='o', markersize=3)
        
        # Plot ground-truth future
        ax.plot(gt_pred_ft[:, 0], gt_pred_ft[:, 1], -gt_pred_ft[:, 2], label='Ground Truth Future', color='#7E2F8E', linewidth=2.5, linestyle='--')
        
        # Plot RADiff predicted future
        ax.plot(pred_ft[:, 0], pred_ft[:, 1], -pred_ft[:, 2], label='RADiff Diffusion Prediction', color='#D95319', linewidth=3.0, linestyle='-')
        
        # Start and End markers
        ax.scatter(obs_ft[0, 0], obs_ft[0, 1], -obs_ft[0, 2], color='green', s=60, label='Start')
        ax.scatter(gt_pred_ft[-1, 0], gt_pred_ft[-1, 1], -gt_pred_ft[-1, 2], color='purple', s=60, label='Target End')
        ax.scatter(pred_ft[-1, 0], pred_ft[-1, 1], -pred_ft[-1, 2], color='red', s=60, label='Pred End')
        
        ax.set_title(f'GUAM UAM Trajectory Sample #{idx+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('North (ft)', fontsize=10)
        ax.set_ylabel('East (ft)', fontsize=10)
        ax.set_zlabel('Altitude / -Down (ft)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.suptitle('NASA GUAM Urban Air Mobility Flight Trajectory Prediction using RADiff Methodology', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Successfully generated trajectory visualization at: {output_path}")
    plt.close()

if __name__ == '__main__':
    visualize_uam_predictions()
