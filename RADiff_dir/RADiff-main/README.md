# RADiff

RADiff is a research-style trajectory prediction repository built around retrieval-augmented diffusion, TrajAirNet-style temporal encoders, graph attention, and CVAE components.

## Repository Layout

```text
RADiff/
├── assets/                 Static files used by visualizations
├── configs/                Experiment configuration files
├── dataset/                Raw and processed trajectory data
├── experiments/            Training and evaluation programs
├── models/                 Model definitions and neural network modules
│   └── diffusion/          Diffusion denoising and 
├── retrieval/              Time-series RAG and embedding utilities
├── saved_models/           Checkpoints and trained weights
├── scripts/                Shell scripts for launching experiments
├── paths.py                Project path helpers
├── train.py                Convenience entry point for training
├── test.py                 Convenience entry point for evaluation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA environment.

## Common Commands

Train:

```bash
python train.py
```

Evaluate:

```bash
python test.py
```

Run training in the background:

```bash
bash scripts/run_train.sh
```

## Main Files

- `experiments/train.py`: training loop and checkpoint saving.
- `experiments/evaluate.py`: evaluation, metrics, and visualization helpers.
- `models/trajairnet.py`: main trajectory prediction model.
- `models/diffusion/`: diffusion backbone and initialization modules.
- `retrieval/`: time-series embedding and similarity search.
- `configs/default.yaml`: baseline experiment settings.

## Notes

- The project intentionally uses a flat research-code layout instead of an installable package layout.
- `models/autoencoder.py` is a legacy module and is not part of the current default training path.
