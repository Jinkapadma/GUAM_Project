import argparse

import os

from datetime import datetime


import numpy as np

from tqdm import tqdm

import torch

from torch.utils.data import DataLoader

from torch import optim


from models.trajairnet import TrajAirNet

from models.utils import TrajectoryDataset, seq_collate_with_padding

from experiments.evaluate import test

from paths import resolve_project_path


import time


def train():

    parser = argparse.ArgumentParser(description="Train TrajAirNet model")

    parser.add_argument("--dataset_folder", type=str, default="/dataset/")

    parser.add_argument("--dataset_name", type=str, default="7days1_small")

    parser.add_argument("--obs", type=int, default=11)

    parser.add_argument("--preds", type=int, default=120)

    parser.add_argument("--preds_step", type=int, default=10)

    parser.add_argument("--points_num", type=int, default=24)

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

    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--total_epochs", type=int, default=20)

    parser.add_argument("--delim", type=str, default=" ")

    parser.add_argument("--evaluate", type=bool, default=True)

    parser.add_argument("--save_model", type=bool, default=True)

    parser.add_argument("--model_pth", type=str, default="/saved_models/")

    args = parser.parse_args()

    print(f"Training dataset: {args.dataset_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datapath = resolve_project_path(
        args.dataset_folder, args.dataset_name, "processed_data"
    )

    print("Loading Train Data from ", os.path.join(datapath, "train"))

    dataset_train = TrajectoryDataset(
        os.path.join(datapath, "train"),
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
    )

    print("Loading Test Data from ", os.path.join(datapath, "test"))

    dataset_test = TrajectoryDataset(
        os.path.join(datapath, "test"),
        obs_len=args.obs,
        pred_len=args.preds,
        step=args.preds_step,
        delim=args.delim,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        collate_fn=seq_collate_with_padding,
    )

    loader_test = DataLoader(
        dataset_test,
        batch_size=16,
        num_workers=4,
        shuffle=True,
        collate_fn=seq_collate_with_padding,
    )

    model = TrajAirNet(args)

    model.to(device)

    print(f"torch.cuda.is_available:{torch.cuda.is_available()}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting Training....")

    for epoch in range(1, args.total_epochs + 1):

        model.train()

        batch_count = 0

        tot_batch_count = 0

        tot_loss = 0

        loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0

        loss_kl = 0

        for batch, embed_searched in tqdm(loader_train):

            a = time.time()

            batch_count += 1

            tot_batch_count += 1

            batch = [tensor.to(device) for tensor in batch]

            (
                obs_traj,
                pred_traj,
                obs_traj_rel,
                pred_traj_rel,
                context,
                seq_start,
            ) = batch

            all_obs_traj_search_results, all_pred_traj_search_results = embed_searched

            num_agents = obs_traj.shape[1]

            adj = torch.ones((num_agents, num_agents))

            optimizer.zero_grad()

            loss_dist, loss_uncertainty, loss_KL = model(
                obs_traj,
                pred_traj,
                adj[0],
                torch.transpose(context, 1, 2),
                all_obs_traj_search_results,
                all_pred_traj_search_results,
            )

            loss = loss_dist * 50 + loss_uncertainty + loss_KL

            loss_total += loss.item()

            loss_dt += loss_dist.item() * 50

            loss_dc += loss_uncertainty.item()

            loss_kl += loss_KL.item()

            loss.backward()

            optimizer.step()

            count += 1

        print(
            "[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}\tLoss KL: {:.6f}".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                epoch,
                loss_total / count,
                loss_dt / count,
                loss_dc / count,
                loss_kl / count,
            )
        )

        if args.save_model:

            loss = tot_loss / tot_batch_count

            model_dir = resolve_project_path(args.model_pth)

            os.makedirs(model_dir, exist_ok=True)

            model_path = os.path.join(
                model_dir, "model_" + args.dataset_name + "_" + str(epoch) + ".pt"
            )

            print("Saving model at", model_path)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "args": args,
                },
                model_path,
            )

        if epoch % 5 == 0:

            print("Starting Testing....")

            model.eval()

            test_ade_loss, test_fde_loss = test(model, loader_test, device)

            print(
                "EPOCH: ",
                epoch,
                "Train Loss: ",
                loss,
                "Test ADE Loss: ",
                test_ade_loss,
                "Test FDE Loss: ",
                test_fde_loss,
            )


if __name__ == "__main__":

    train()
