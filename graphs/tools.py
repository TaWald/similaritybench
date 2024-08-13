import random

import numpy as np
import torch


def subsample_torch_mask(idx, size, seed):
    # sample from numpy random state for more safe reproducibility
    rng = np.random.RandomState(seed)
    numeric_idx = np.arange(len(idx))[idx.numpy()]

    sub_idx = torch.from_numpy(rng.choice(numeric_idx, size, replace=False))
    res_idx = torch.zeros(size=(len(idx),), dtype=torch.bool)
    res_idx[sub_idx] = True

    return res_idx


def subsample_torch_index(idx, size, seed):
    # sample from numpy random state for more safe reproducibility
    rng = np.random.RandomState(seed)

    return torch.from_numpy(rng.choice(idx.numpy(), size, replace=False))


def shuffle_labels(y, frac=0.5, seed=None):
    if seed is not None:
        random.seed(seed)

    is_tensor = torch.is_tensor(y)

    if is_tensor:
        y = y.numpy().flatten()

    n_instances = len(y)
    Y = list(np.unique(y))
    shuffle_idx = random.sample(list(range(n_instances)), k=int(frac * n_instances))
    for i in shuffle_idx:
        old_label = y[i]
        new_label = random.sample([label for label in Y if label != old_label], k=1)[0]
        y[i] = new_label

    if is_tensor:
        return torch.from_numpy(np.reshape(y, newshape=(n_instances, 1)))

    return y


# ---------------------------------------------- P-GNN HELPER FUNCTIONS ------------------------------------------------
# code taken from https://github.com/JiaxuanYou/P-GNN/blob/master/utils.py
# ----------------------------------------------------------------------------------------------------------------------


def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    copy = int(c * m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n / np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchorset_id


def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0], len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0], len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:, i] = dist_max_temp
        dist_argmax[:, i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device="cpu"):
    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num // anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2 ** (i + 1) - 1
        anchors = np.random.choice(data.num_nodes, size=(layer_num, anchor_num_per_size, anchor_size), replace=True)
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes, c=1)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)
