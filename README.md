# GUAM Project: RADiff for UAM Trajectory Prediction

This repository contains a retrieval-augmented diffusion framework for predicting future urban air mobility (UAM) trajectories from the NASA GUAM dataset. The project integrates dataset parsing, retrieval-based context modeling, diffusion-based forecasting, evaluation, and output visualization.

## Recent updates

The current implementation includes:
- GUAM trajectory parsing and normalization from `.mat` input files
- time-series embedding and nearest-neighbor retrieval using FAISS or scikit-learn fallback
- a transformer-based diffusion model for future trajectory generation
- training and evaluation code with ADE/FDE metric reporting
- checkpoint saving to `saved_models/`
- 3D visualization outputs for observed history, ground truth, and model prediction
- a command-line execution entry point via `run_pipeline.py`

## Repository structure

- `guam_parser.py` — parses and splits GUAM trajectories into observation and prediction windows
- `rag_retriever.py` — embeds trajectories and retrieves similar historical motion patterns
- `radiff_model.py` — contains the diffusion model and reverse-sampling generation routine
- `train_eval_guam.py` — trains the RADiff model, computes metrics, and saves checkpoints
- `visualize_results.py` — generates 3D trajectory plots for comparison
- `run_pipeline.py` — runs the full workflow end-to-end
- `saved_models/` — trained model checkpoints and evaluation artifacts
- `GUAM_dir/` — GUAM source folder and related MATLAB assets
- `README.md` — project overview and setup guide

## Environment setup

Install the required Python packages:

```bash
pip install numpy matplotlib h5py scipy torch faiss-cpu
```

If FAISS installation is unavailable in your environment, the project contains a fallback using `NearestNeighbors` from scikit-learn in `rag_retriever.py`.

## Dataset configuration

The project expects the GUAM dataset file at a known local path. The current default path in the code is:

```python
r'd:\New folder\GUAM_dir\Generic-Urban-Air-Mobility-GUAM-main\Challenge_Problems\Data_Set_1.mat'
```

If your workspace is stored in a different location, update the dataset path before running the pipeline.

## Running the project

From the project root:

```bash
python run_pipeline.py
```

This performs the following steps:
1. load and preprocess GUAM trajectory data
2. build the retrieval index
3. train the RADiff model
4. evaluate predictions using ADE/FDE
5. save a checkpoint and generate a visualization image

## Outputs

The pipeline produces:
- model checkpoint files in `saved_models/`
- evaluation values printed to the console
- a trajectory visualization PNG such as `uam_trajectory_prediction.png`

## Git and branch setup

To push this project to the GitHub branch named `Padma`, use:

```bash
git init
git remote add origin https://github.com/Jinkapadma/GUAM_Project.git
git checkout -b Padma
git add .
git commit -m "Initial project upload"
git push -u origin Padma
```

If Git throws an identity error, configure your local Git user:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Then repeat the commit and push commands.

## Notes

- The project uses absolute Windows paths in the default configuration, so local path changes may be necessary on other machines.
- The repository includes the full GUAM Matlab folder under `GUAM_dir/` for reference and compatibility.
- This project is intended for research, experimentation, and prototype development.

## License

Please check the repository license and dataset use conditions before publishing or sharing results externally.
