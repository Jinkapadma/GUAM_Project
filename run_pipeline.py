import os
import sys
from train_eval_guam import train_and_evaluate
from visualize_results import visualize_uam_predictions

def main():
    print("=" * 70)
    print("  NASA GUAM Trajectory Prediction using RADiff Methodology")
    print("=" * 70)
    
    mat_file = r'd:\New folder\GUAM_dir\Generic-Urban-Air-Mobility-GUAM-main\Challenge_Problems\Data_Set_1.mat'
    if not os.path.exists(mat_file):
        print(f"Error: GUAM dataset file not found at {mat_file}")
        sys.exit(1)
        
    print("\n[Step 1/3] Parsing GUAM trajectories, building FAISS RAG, and training RADiff model...")
    checkpoint_path, dataset_dict, pred_test_norm = train_and_evaluate(
        mat_file=mat_file,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        obs_len=12,
        pred_len=24,
        max_trajectories=1500,
        k_neighbors=5,
        save_dir='saved_models'
    )
    
    print("\n[Step 2/3] Generating 3D UAM Flight Trajectory Visualizations...")
    viz_path = 'uam_trajectory_prediction.png'
    visualize_uam_predictions(
        checkpoint_path=checkpoint_path,
        output_path=viz_path,
        sample_indices=[0, 1, 2]
    )
    
    print("\n[Step 3/3] Execution Complete!")
    print(f" - Model Checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f" - Visualization Artifact: {os.path.abspath(viz_path)}")
    print("=" * 70)

if __name__ == '__main__':
    main()
