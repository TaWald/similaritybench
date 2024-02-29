import random

import numpy as np
import torch


def shuffle_labels(y, frac=0.5):

    b_tensor = torch.is_tensor(y)

    if b_tensor:
        y = y.numpy().flatten()

    n_instances = len(y)
    Y = list(np.unique(y))
    shuffle_idx = random.sample(list(range(n_instances)), k=int(frac * n_instances))
    for i in shuffle_idx:
        old_label = y[i]
        new_label = random.sample([label for label in Y if label != old_label], k=1)[0]
        y[i] = new_label

    if b_tensor:
        return torch.from_numpy(np.reshape(y, (n_instances, 1)))

    return y
