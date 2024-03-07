import random
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as func
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.conv import SAGEConv
from torcheval.metrics.functional import multiclass_accuracy
from tqdm import tqdm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = func.relu(x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = func.relu(x)
            x = func.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = func.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def train_model(model, data, split_idx, seed: int, optimizer_params: Dict, save_path: str, b_test: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)
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

        epoch_res = [epoch, loss, train_acc, val_acc]
        if b_test:
            test_acc = test(model, data, test_idx=split_idx["test"])
            epoch_res.append(test_acc)

        results.append(epoch_res)

    torch.save(model.state_dict(), save_path)

    if b_test:
        return results, test(model, data, split_idx["test"])

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

    train_acc, val_acc = (
        multiclass_accuracy(train_pred, data.y[train_idx].squeeze(1)).detach().cpu().numpy(),
        multiclass_accuracy(val_pred, data.y[val_idx].squeeze(1)).detach().cpu().numpy(),
    )

    return train_acc, val_acc


@torch.no_grad()
def test(model, data, test_idx):
    model.eval()

    out = model(data.x, data.adj_t)[test_idx]
    pred = out.argmax(dim=-1, keepdim=True)

    return multiclass_accuracy(pred, data.y.squeeze(1)[test_idx]).detach().cpu().numpy()


@torch.no_grad()
def get_representations(model, data, test_idx, layer_ids):
    model.eval()

    activations = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    hooks = dict()
    for i in layer_ids:
        hooks[i] = model.convs[i].register_forward_hook(getActivation(f"layer{i + 1}"))

    _ = model(data.x, data.adj_t)

    for i in layer_ids:
        hooks[i].remove()

    reps = dict()
    for i in layer_ids:
        reps[i] = activations[f"layer{i + 1}"].detach()[test_idx].numpy()

    return reps
