import argparse

import os

import time


from tqdm import tqdm

import numpy as np


import torch

from torch.utils.data import DataLoader


from models.trajairnet import TrajAirNet

from models.utils import ade, fde, TrajectoryDataset, seq_collate_with_padding

from paths import project_path, resolve_project_path

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from collections import OrderedDict

from matplotlib.colors import to_rgb

from scipy.stats import gaussian_kde


from scipy import stats

import warnings

warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser(description="Test TrajAirNet model")

    parser.add_argument("--dataset_folder", type=str, default="/dataset/")

    parser.add_argument("--dataset_name", type=str, default="7days1_small")

    parser.add_argument("--epoch", type=int, default=20)

    parser.add_argument("--obs", type=int, default=11)

    parser.add_argument("--preds", type=int, default=120)

    parser.add_argument("--preds_step", type=int, default=10)

    parser.add_argument("--input_channels", type=int, default=3)

    parser.add_argument("--tcn_channel_size", type=int, default=256)

    parser.add_argument("--tcn_layers", type=int, default=2)

    parser.add_argument("--tcn_kernels", type=int, default=4)

    parser.add_argument("--num_context_input_c", type=int, default=2)

    parser.add_argument("--num_context_output_c", type=int, default=7)

    parser.add_argument("--cnn_kernels", type=int, default=2)

    parser.add_argument("--gat_heads", type=int, default=16)

    parser.add_argument("--graph_hidden", type=int, default=256)

    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--cvae_hidden", type=int, default=128)

    parser.add_argument("--cvae_channel_size", type=int, default=128)

    parser.add_argument("--cvae_layers", type=int, default=2)

    parser.add_argument("--mlp_layer", type=int, default=32)

    parser.add_argument("--delim", type=str, default=" ")

    parser.add_argument("--model_dir", type=str, default="/saved_models/111days/")

    parser.add_argument("--k", type=int, default=4)

    parser.add_argument("--num_samples", type=int, default=20)

    parser.add_argument("--traj_dim", type=int, default=3)

    parser.add_argument("--agent_num", type=int, default=3)

    args = parser.parse_args()

    print(f"Evaluation dataset: {args.dataset_name}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datapath = resolve_project_path(
        args.dataset_folder, args.dataset_name, "processed_data"
    )

    print("Loading Test Data from ", os.path.join(datapath, "test"))

    dataset_test = TrajectoryDataset(
        os.path.join(datapath, "test"),
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
    )

    loader_test = DataLoader(
        dataset_test,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        collate_fn=seq_collate_with_padding,
    )

    model = TrajAirNet(args)

    model.to(device)

    model_path = os.path.join(
        resolve_project_path(args.model_dir), f"model_111_days_90.pt"
    )

    print(f"Loading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    test_ade_loss, test_fde_loss = test(model, loader_test, device)

    print("Test ADE Loss: ", test_ade_loss, "Test FDE Loss: ", test_fde_loss)


def test(model, loader_test, device):

    tot_ade_loss = 0

    tot_fde_loss = 0

    tot_batch = 0

    tot_scene = 0

    for batch, embed_searched in tqdm(loader_test):

        tot_batch += 1

        batch = [tensor.to(device) for tensor in batch]

        (
            obs_traj_all,
            pred_traj_all,
            obs_traj_rel_all,
            pred_traj_rel_all,
            context,
            seq_start,
        ) = batch

        batch_size = obs_traj_all.shape[0]

        all_obs_traj_search_results, all_pred_traj_search_results = embed_searched

        num_agents = obs_traj_all.shape[1]

        adj = torch.ones((num_agents, num_agents))

        best_ade_loss = float("inf")

        best_fde_loss = float("inf")

        recon_y_all = model.inference(
            obs_traj_all,
            pred_traj_all,
            adj[0],
            torch.transpose(context, 1, 2),
            all_obs_traj_search_results,
            all_pred_traj_search_results,
        )

        recon_y_all = torch.reshape(
            recon_y_all,
            (
                batch_size,
                num_agents,
                recon_y_all.shape[1],
                recon_y_all.shape[2],
                recon_y_all.shape[3],
            ),
        )

        ade_loss = 0

        fde_loss = 0

        all_agent_num = 0

        all_min_ade = {}

        all_min_fde = {}

        number_ade_over_fde = 0

        total = 0

        ade_fde_error = 0

        ade_fde_error_list = 0

        for bs in range(batch_size):

            tot_scene += 1

            scene_ade_loss = 0

            scene_fde_loss = 0

            new_num_agents = 1

            new_obs_traj_all = obs_traj_all[bs].clone().cpu().numpy()

            for dup in range(1, num_agents):

                if new_obs_traj_all[0][0][0] == new_obs_traj_all[dup][0][0]:

                    new_num_agents = dup

                    break

            all_agent_num += new_num_agents

            plt.figure(figsize=(10, 8), dpi=150)

            for agent in range(new_num_agents):

                obs_traj = np.squeeze(obs_traj_all[bs, agent, :, :].cpu().numpy())

                pred_traj = np.squeeze(pred_traj_all[bs, agent, :, :].cpu().numpy())

                recon_pred = (
                    recon_y_all[bs, agent, :, :, :].detach().cpu().numpy().transpose()
                )

                min_ade_loss = float("inf")

                min_fde_loss = float("inf")

                for k in range(recon_pred.shape[2]):

                    single_ade_loss = ade(recon_pred[:, :, k], pred_traj)

                    single_fde_loss = fde((recon_pred[:, :, k]), (pred_traj))

                    if single_ade_loss <= min_ade_loss:

                        min_ade_loss = single_ade_loss

                        min_fde_loss = single_fde_loss

                        n = k

                ade_loss += min_ade_loss

                fde_loss += min_fde_loss

                total += 1

                if min_ade_loss >= min_fde_loss:

                    number_ade_over_fde += 1

                    ade_fde_error += min_ade_loss - min_fde_loss

        ade_total_loss = ade_loss / all_agent_num

        fde_total_loss = fde_loss / all_agent_num

        if ade_total_loss < best_ade_loss:

            best_ade_loss = ade_total_loss

            best_fde_loss = fde_total_loss

        tot_ade_loss += best_ade_loss

        tot_fde_loss += best_fde_loss

    return tot_ade_loss / (tot_batch), tot_fde_loss / (tot_batch)


if __name__ == "__main__":

    main()
