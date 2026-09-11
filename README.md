# AIRRADIFF Project Commands

This guide contains the terminal commands to run the AIRRADIFF project to get the results one by one. You can run these at any time in your VS Code terminal, PowerShell, or Command Prompt.

## 1. Get the ADE and FDE Evaluation Results
*(This will train/evaluate the model and print the metrics to your screen)*
```bash
python train_eval_guam.py
```

## 2. Generate the 3D Trajectory Prediction Image
*(This will create the `uam_trajectory_prediction.png` visual plot)*
```bash
python visualize_results.py
```

## 3. Extract Metrics from a Saved Model (Optional)
*(If you want to print metrics instantly from the saved model without re-training)*
```bash
python -c "import torch; ckpt = torch.load('saved_models/radiff_guam_model.pt', map_location='cpu', weights_only=False); print('Normalized ADE:', ckpt.get('ade_norm'), '| Normalized FDE:', ckpt.get('fde_norm')); print('Real Scale ADE:', ckpt.get('ade_ft'), 'ft | Real Scale FDE:', ckpt.get('fde_ft'), 'ft')"
```

## 4. Run the Complete Pipeline
*(This will execute all steps in one go)*
```bash
python run_pipeline.py
```
