import random

from sklearn.metrics import accuracy_score

import numpy as np
import torch
from torch.nn import functional as func

from tqdm import tqdm
from typing import Dict


def train_model(model, data, split_idx, seed: int, optimizer_params: Dict, save_path: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    data = data.to(device)

    # TODO: maybe rearrange this, data passing currently quite messy
    train_idx = split_idx["train"].to(device)
    val_idx = split_idx["valid"]

    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params["lr"])

    results = []
    for epoch in tqdm(range(1, 1 + optimizer_params["epochs"])):
        loss = train_epoch(model, data, train_idx, optimizer)
        train_acc, val_acc = validate(model, data, train_idx, val_idx)
        results.append((epoch, loss, train_acc, val_acc))

    torch.save(model.state_dict(), save_path)

    return results


def train_epoch(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = func.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def validate(model, data, train_idx, val_idx):
    model.eval()

    out_train = model(data.x, data.adj_t)[train_idx]
    train_pred = out_train.argmax(dim=-1, keepdim=True)

    out_val = model(data.x, data.adj_t)[val_idx]
    val_pred = out_val.argmax(dim=-1, keepdim=True)

    train_acc, val_acc = accuracy_score(data.y[train_idx], train_pred), accuracy_score(data.y[val_idx], val_pred)

    return train_acc, val_acc


@torch.no_grad()
def get_representations(model, data, test_idx, n_layers):
    model.eval()

    activations = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    hooks = dict()
    for i in range(n_layers):
        hooks[i] = model.convs[i].register_forward_hook(getActivation(f"layer{i + 1}"))

    _ = model(data.x, data.adj_t)[test_idx]

    for i in range(n_layers):
        hooks[i].remove()

    reps = dict()
    for i in range(n_layers):
        reps[i] = activations[f"layer{i + 1}"].detach().numpy()

    return reps
