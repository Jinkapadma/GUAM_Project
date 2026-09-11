import time

import matplotlib.pyplot as plt

import math

import os

from tqdm import tqdm


from torch import nn

import torch

from torch.utils.data import Dataset


import numpy as np

from scipy.spatial import distance_matrix

import random


from retrieval.Rag import TimeSeriesRAG, TimeSeriesDocument

import uuid

from retrieval.Rag_embedder import TimeSeriesEmbedder

from paths import project_path


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets
    Modified from https://github.com/alexmonti19/dagnet"""

    def __init__(
        self,
        data_dir,
        obs_len=11,
        pred_len=120,
        skip=8,
        step=10,
        min_agent=0,
        delim=" ",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <agent_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_agent: Minimum number of agents that should be in a seqeunce
        - step: Subsampling for pred
        - delim: Delimiter in the dataset files
        """

        super(TrajectoryDataset, self).__init__()

        self.max_agents_in_frame = 0

        self.data_dir = data_dir

        self.obs_len = obs_len

        self.pred_len = pred_len

        self.skip = skip

        self.step = step

        self.seq_len = self.obs_len + self.pred_len

        self.delim = delim

        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len / self.step))

        self.embedder = TimeSeriesEmbedder()

        all_files = os.listdir(self.data_dir)

        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]

        num_agents_in_seq = []

        seq_list = []

        seq_list_rel = []

        context_list = []

        for path in tqdm(all_files):

            data = read_file(path, delim)

            if len(data) == 0 or len(data[:, 0]) == 0:

                print("File is empty")

                continue

            if len(data) < (self.obs_len + self.pred_len):

                continue

            frames = np.unique(data[:, 0]).tolist()

            frame_data = []

            for frame in frames:

                frame_data.append(data[frame == data[:, 0], :])

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):

                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )

                agents_in_curr_seq = np.unique(curr_seq_data[:, 1])

                self.max_agents_in_frame = max(
                    self.max_agents_in_frame, len(agents_in_curr_seq)
                )

                curr_seq_rel = np.zeros(
                    (len(agents_in_curr_seq), 3, self.seq_final_len)
                )

                curr_seq = np.zeros((len(agents_in_curr_seq), 3, self.seq_final_len))

                curr_context = np.zeros(
                    (len(agents_in_curr_seq), 2, self.seq_final_len)
                )

                num_agents_considered = 0

                for _, agent_id in enumerate(agents_in_curr_seq):

                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] == agent_id, :]

                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx

                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:

                        continue

                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])

                    obs = curr_agent_seq[:, :obs_len]

                    pred = curr_agent_seq[:, obs_len + step - 1 :: step]

                    curr_agent_seq = np.hstack((obs, pred))

                    context = curr_agent_seq[-2:, :]

                    assert ~np.isnan(context).any()

                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)

                    rel_curr_agent_seq[:, 1:] = (
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                    )

                    _idx = num_agents_considered

                    if curr_agent_seq.shape[1] != self.seq_final_len:

                        continue

                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3, :]

                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[:3, :]

                    curr_context[_idx, :, pad_front:pad_end] = context

                    num_agents_considered += 1

                if num_agents_considered > min_agent:

                    num_agents_in_seq.append(num_agents_considered)

                    seq_list.append(curr_seq[:num_agents_considered])

                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])

                    context_list.append(curr_context[:num_agents_considered])

        self.num_seq = len(seq_list)

        seq_list = np.concatenate(seq_list, axis=0)

        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        context_list = np.concatenate(context_list, axis=0)

        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )

        self.obs_context = torch.from_numpy(context_list[:, :, : self.obs_len]).type(
            torch.float
        )

        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )

        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )

        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )

        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()

        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        self.max_agents = -float("Inf")

        for start, end in self.seq_start_end:

            n_agents = end - start

            self.max_agents = (
                n_agents if n_agents > self.max_agents else self.max_agents
            )

        self.obs_traj_embed = self.embedder.embed_batch(
            np.transpose(self.obs_traj.clone().cpu().numpy(), (0, 2, 1))
        )

        self.pred_traj_embed = self.embedder.embed_batch(
            np.transpose(self.pred_traj.clone().cpu().numpy(), (0, 2, 1))
        )

        data_dir_list = [project_path("dataset", "rag_knowledge")]

        print(f"RAG knowledge sources: {data_dir_list}")

        rag = TrajectoryDataset_RAG(
            data_dir_list=data_dir_list,
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            step=self.step,
            delim=" ",
        ).rag_system

        a = time.time()

        topk = 20

        self.obs_traj_search_results = rag.search_batch(self.obs_traj_embed, k=topk)

        print(f"RAG retrieval top-k: {topk}")

        b = time.time()

        print(f"RAG retrieval time: {(b - a) / 60:.2f} minutes")

        self.pred_traj_search_results = []

    def __len__(self):

        return self.num_seq

    def __max_agents__(self):

        return self.max_agents

    def __getitem__(self, index):

        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.obs_context[start:end, :],
            self.obs_traj_search_results[start:end],
            self.pred_traj_search_results[start:end],
        ]

        return out


class DotDict(dict):
    r"""dot.notation access to dictionary attributes"""

    __getattr__ = dict.get

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__

    __getstate__ = dict

    __setstate__ = dict.update


class TrajectoryDataset_RAG(Dataset):
    """Dataloder for the Trajectory datasets
    Modified from https://github.com/alexmonti19/dagnet"""

    def __init__(
        self,
        data_dir_list,
        obs_len=11,
        pred_len=120,
        skip=8,
        step=10,
        min_agent=0,
        delim=" ",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <agent_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - min_agent: Minimum number of agents that should be in a seqeunce
        - step: Subsampling for pred
        - delim: Delimiter in the dataset files
        """

        super(TrajectoryDataset_RAG, self).__init__()

        self.max_agents_in_frame = 0

        self.data_dir_list = data_dir_list

        self.obs_len = obs_len

        self.pred_len = pred_len

        self.skip = skip

        self.step = step

        self.seq_len = self.obs_len + self.pred_len

        self.delim = delim

        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len / self.step))

        self.rag_system = TimeSeriesRAG()

        for data_dir in data_dir_list:

            all_files = os.listdir(data_dir)

            all_files = [os.path.join(data_dir, _path) for _path in all_files]

            num_agents_in_seq = []

            seq_list = []

            seq_list_rel = []

            context_list = []

            for path in tqdm(all_files):

                data = read_file(path, delim)

                if len(data[:, 0]) == 0:

                    print("File is empty")

                    continue

                frames = np.unique(data[:, 0]).tolist()

                frame_data = []

                for frame in frames:

                    frame_data.append(data[frame == data[:, 0], :])

                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):

                    curr_seq_data = np.concatenate(
                        frame_data[idx : idx + self.seq_len], axis=0
                    )

                    agents_in_curr_seq = np.unique(curr_seq_data[:, 1])

                    self.max_agents_in_frame = max(
                        self.max_agents_in_frame, len(agents_in_curr_seq)
                    )

                    curr_seq_rel = np.zeros(
                        (len(agents_in_curr_seq), 3, self.seq_final_len)
                    )

                    curr_seq = np.zeros(
                        (len(agents_in_curr_seq), 3, self.seq_final_len)
                    )

                    curr_context = np.zeros(
                        (len(agents_in_curr_seq), 2, self.seq_final_len)
                    )

                    num_agents_considered = 0

                    for _, agent_id in enumerate(agents_in_curr_seq):

                        curr_agent_seq = curr_seq_data[
                            curr_seq_data[:, 1] == agent_id, :
                        ]

                        pad_front = frames.index(curr_agent_seq[0, 0]) - idx

                        pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1

                        if pad_end - pad_front != self.seq_len:

                            continue

                        curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])

                        obs = curr_agent_seq[:, :obs_len]

                        pred = curr_agent_seq[:, obs_len + step - 1 :: step]

                        curr_agent_seq = np.hstack((obs, pred))

                        context = curr_agent_seq[-2:, :]

                        assert ~np.isnan(context).any()

                        rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)

                        rel_curr_agent_seq[:, 1:] = (
                            curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]
                        )

                        _idx = num_agents_considered

                        if curr_agent_seq.shape[1] != self.seq_final_len:

                            continue

                        curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3, :]

                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[
                            :3, :
                        ]

                        curr_context[_idx, :, pad_front:pad_end] = context

                        num_agents_considered += 1

                    if num_agents_considered > min_agent:

                        num_agents_in_seq.append(num_agents_considered)

                        seq_list.append(curr_seq[:num_agents_considered])

                        seq_list_rel.append(curr_seq_rel[:num_agents_considered])

                        context_list.append(curr_context[:num_agents_considered])

        self.num_seq = len(seq_list)

        seq_list = np.concatenate(seq_list, axis=0)

        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        context_list = np.concatenate(context_list, axis=0)

        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )

        self.obs_context = torch.from_numpy(context_list[:, :, : self.obs_len]).type(
            torch.float
        )

        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )

        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )

        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )

        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()

        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        self.max_agents = -float("Inf")

        for start, end in self.seq_start_end:

            n_agents = end - start

            self.max_agents = (
                n_agents if n_agents > self.max_agents else self.max_agents
            )

        obs_traj = seq_list[:, :, : self.obs_len]

        pred_traj = seq_list[:, :, self.obs_len :]

        embedder = TimeSeriesEmbedder()

        for i in range(seq_list.shape[0]):

            obs_sub_traj = obs_traj[i]

            pred_sub_traj = pred_traj[i]

            metadata_dict = {}

            doc_id = str(uuid.uuid4())

            obs_sub_traj = np.transpose(obs_sub_traj, (1, 0))

            embedding = embedder.embed(obs_sub_traj)

            doc = TimeSeriesDocument(
                id=doc_id,
                data=obs_sub_traj,
                pred_data=np.transpose(pred_sub_traj, (1, 0)),
                metadata=metadata_dict,
                embedding=embedding,
            )

            self.rag_system.add_document(doc)


def ade(y1, y2):
    """
    y: (seq_len,2)
    """

    if y1.shape[0] == 3:

        y1 = np.transpose(y1, (1, 0))

        y2 = np.transpose(y2, (1, 0))

    loss = y1 - y2

    loss = loss**2

    loss = np.sqrt(np.sum(loss, 1))

    return np.mean(loss)


def fde(y1, y2):

    if y1.shape[0] == 3:

        loss = (y1[:, -1] - y2[:, -1]) ** 2

    else:

        loss = (y1[-1, :] - y2[-1, :]) ** 2

    return np.sqrt(np.sum(loss))


def rel_to_abs(obs, rel_pred):

    pred = rel_pred.copy()

    pred[0] += obs[-1]

    for i in range(1, len(pred)):

        pred[i] += pred[i - 1]

    return pred


def rmse(y1, y2):

    criterion = nn.MSELoss()

    return torch.sqrt(criterion(y1, y2))


def read_file(_path, delim="\t"):

    data = []

    if delim == "tab":

        delim = "\t"

    elif delim == "space":

        delim = " "

    with open(_path, "r") as f:

        for line in f:

            line = line.strip().split(delim)

            line = [float(i) for i in line]

            data.append(line)

    return np.asarray(data)


def acc_to_abs(acc, obs, delta=1):

    acc = acc.permute(2, 1, 0)

    pred = torch.empty_like(acc)

    pred[0] = 2 * obs[-1] - obs[0] + acc[0]

    pred[1] = 2 * pred[0] - obs[-1] + acc[1]

    for i in range(2, acc.shape[0]):

        pred[i] = 2 * pred[i - 1] - pred[i - 2] + acc[i]

    return pred.permute(2, 1, 0)


def seq_collate(data):

    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, context_list) = (
        zip(*data)
    )

    _len = [len(seq) for seq in obs_seq_list]

    cum_start_idx = [0] + np.cumsum(_len).tolist()

    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)

    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)

    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)

    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)

    context = torch.cat(context_list, dim=0).permute(2, 0, 1)

    seq_start_end = torch.LongTensor(seq_start_end)

    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, context, seq_start_end]

    return tuple(out)


def seq_collate_with_padding(data):

    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        context_list,
        obs_traj_search_results,
        pred_traj_search_results,
    ) = zip(*data)

    padding_num = 5

    new_obs_seq_list = []

    new_pred_seq_list = []

    new_obs_seq_rel_list = []

    new_pred_seq_rel_list = []

    new_context_list = []

    new_obs_traj_embed_list = []

    new_pred_traj_embed_list = []

    new_distrance_embed_list = []

    for i in range(len(data)):

        agent_num = obs_seq_list[i].shape[0]

        agent_limit = 5

        if agent_num > agent_limit:

            print(f"Skip sequence with too many agents: {agent_num}")

            continue

            obs_seq_list[i] = obs_seq_list[i][:padding_num, :, :]

        index_list = list(range(agent_num))

        index_list = index_list * agent_limit

        need_padding = padding_num - agent_num

        index_list = index_list[:need_padding]

        obs_traj_padding_aircraft_list = []

        pred_traj_padding_aircraft_list = []

        obs_traj_rel_padding_aircraft_list = []

        pred_traj_rel_padding_aircraft_list = []

        context_padding_aircraft_list = []

        obs_traj_embed_padding_aircraft_list = []

        pred_traj_embed_padding_aircraft_list = []

        distance_padding_aircraft_list = []

        for index in index_list:

            obs_traj_padding_aircraft = obs_seq_list[i][index].detach().clone()

            pred_traj_padding_aircraft = pred_seq_list[i][index].detach().clone()

            obs_traj_rel_padding_aircraft = obs_seq_rel_list[i][index].detach().clone()

            pred_traj_rel_padding_aircraft = (
                pred_seq_rel_list[i][index].detach().clone()
            )

            context_padding_aircraft = context_list[i][index].detach().clone()

            obs_traj_embed_padding_aircraft = obs_traj_search_results[i][index].copy()

            obs_traj_padding_aircraft_list.append(
                obs_traj_padding_aircraft.unsqueeze(dim=0)
            )

            pred_traj_padding_aircraft_list.append(
                pred_traj_padding_aircraft.unsqueeze(dim=0)
            )

            obs_traj_rel_padding_aircraft_list.append(
                obs_traj_rel_padding_aircraft.unsqueeze(dim=0)
            )

            pred_traj_rel_padding_aircraft_list.append(
                pred_traj_rel_padding_aircraft.unsqueeze(dim=0)
            )

            context_padding_aircraft_list.append(
                context_padding_aircraft.unsqueeze(dim=0)
            )

            pred_data = [vo["pred_data"] for vo in obs_traj_embed_padding_aircraft]

            pred_data = np.array(pred_data)

            obs_traj_embed_padding_aircraft_list.append(pred_data)

            distance = [vo["distance"] for vo in obs_traj_embed_padding_aircraft]

            distance = np.array(distance)

            distance_padding_aircraft_list.append(distance)

        if len(obs_traj_padding_aircraft_list) != 0:

            _obs_traj_padding_aircraft = torch.cat(
                obs_traj_padding_aircraft_list, dim=0
            )

            _pred_traj_padding_aircraft = torch.cat(
                pred_traj_padding_aircraft_list, dim=0
            )

            _obs_traj_rel_padding_aircraft = torch.cat(
                obs_traj_rel_padding_aircraft_list, dim=0
            )

            _pred_traj_rel_padding_aircraft = torch.cat(
                pred_traj_rel_padding_aircraft_list, dim=0
            )

            _context_padding_aircraft = torch.cat(context_padding_aircraft_list, dim=0)

            _obs_traj_embed_padding_aircraft = np.array(
                obs_traj_embed_padding_aircraft_list
            )

            _distance_padding_aircraft = np.array(distance_padding_aircraft_list)

            new_obs_seq_list.append(
                torch.cat([obs_seq_list[i], _obs_traj_padding_aircraft], dim=0)
            )

            new_pred_seq_list.append(
                torch.cat([pred_seq_list[i], _pred_traj_padding_aircraft], dim=0)
            )

            new_obs_seq_rel_list.append(
                torch.cat([obs_seq_rel_list[i], _obs_traj_rel_padding_aircraft], dim=0)
            )

            new_pred_seq_rel_list.append(
                torch.cat(
                    [pred_seq_rel_list[i], _pred_traj_rel_padding_aircraft], dim=0
                )
            )

            new_context_list.append(
                torch.cat([context_list[i], _context_padding_aircraft], dim=0)
            )

            all_ori_pred_data = []

            all_ori_distance = []

            for ori_index in range(len(obs_traj_search_results[i])):

                ori_pred_data = [
                    vo["pred_data"] for vo in obs_traj_search_results[i][ori_index]
                ]

                ori_pred_data = np.array(ori_pred_data)

                all_ori_pred_data.append(ori_pred_data)

                ori_distance = [
                    vo["distance"] for vo in obs_traj_search_results[i][ori_index]
                ]

                ori_distance = np.array(ori_distance)

                all_ori_distance.append(ori_distance)

            all_ori_pred_data = np.array(all_ori_pred_data)

            new_obs_traj_embed_list.append(
                np.append(all_ori_pred_data, _obs_traj_embed_padding_aircraft, axis=0)
            )

            all_ori_distance = np.array(all_ori_distance)

            new_distrance_embed_list.append(
                np.append(all_ori_distance, _distance_padding_aircraft, axis=0)
            )

        else:

            new_obs_seq_list.append(obs_seq_list[i])

            new_pred_seq_list.append(pred_seq_list[i])

            new_obs_seq_rel_list.append(obs_seq_rel_list[i])

            new_pred_seq_rel_list.append(pred_seq_rel_list[i])

            new_context_list.append(context_list[i])

            all_ori_pred_data = []

            all_ori_distance = []

            for ori_index in range(len(obs_traj_search_results[i])):

                ori_pred_data = [
                    vo["pred_data"] for vo in obs_traj_search_results[i][ori_index]
                ]

                ori_pred_data = np.array(ori_pred_data)

                all_ori_pred_data.append(ori_pred_data)

                ori_distance = [
                    vo["distance"] for vo in obs_traj_search_results[i][ori_index]
                ]

                ori_distance = np.array(ori_distance)

                all_ori_distance.append(ori_distance)

            all_ori_pred_data = np.array(all_ori_pred_data)

            new_obs_traj_embed_list.append(all_ori_pred_data)

            all_ori_distance = np.array(all_ori_distance)

            new_distrance_embed_list.append(all_ori_distance)

    seq_start_end = torch.tensor([1, 2, 3, 4, 5])

    obs_traj = torch.stack(new_obs_seq_list, dim=0)

    pred_traj = torch.stack(new_pred_seq_list, dim=0)

    obs_traj_rel = torch.stack(new_obs_seq_rel_list, dim=0)

    pred_traj_rel = torch.stack(new_pred_seq_rel_list, dim=0)

    context = torch.stack(new_context_list, dim=0)

    searched_results = np.array(new_obs_traj_embed_list)

    searched_distance = np.array(new_distrance_embed_list)

    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        context,
        seq_start_end,
    ]

    return tuple(out), tuple([searched_results, searched_distance])


def loss_func(recon_y, y, mean, log_var):

    traj_loss = rmse(recon_y, y)

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return traj_loss + KLD


def loss_func_MSE(recon_y, y):

    min_loss = 0

    for i in range(recon_y.shape[0]):

        traj_loss = rmse(recon_y[i], y.squeeze())

        min_loss += traj_loss

    return min_loss


def endpoint_smoothing_torch(trajs, lam=1.0):
    """
    trajs: torch.Tensor [N, T, D]
    """

    N, T, D = trajs.shape

    if T <= 2:

        return trajs

    n = T - 2

    device = trajs.device

    A = torch.eye(n, device=device) * (1 + 2 * lam)

    A[:-1, 1:] += torch.diag(torch.full((n - 1,), -lam, device=device))

    A[1:, :-1] += torch.diag(torch.full((n - 1,), -lam, device=device))

    A_inv = torch.linalg.inv(A)

    trajs_smooth = trajs.clone()

    for i in range(N):

        for d in range(D):

            mid = trajs[i, 1:-1, d]

            b = mid.clone()

            b[0] += lam * trajs[i, 0, d]

            b[-1] += lam * trajs[i, -1, d]

            trajs_smooth[i, 1:-1, d] = A_inv @ b

    return trajs_smooth


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """

    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):

        mat_a_i = mat_a[:, i, :, :].squeeze(-2)

        mat_b_i = mat_b[0, i, :, :].squeeze()

        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """

    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1

    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)
