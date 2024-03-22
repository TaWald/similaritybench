import argparse
import copy
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from graphs.config import DATA_DIR
from graphs.config import GNN_DICT
from graphs.config import GNN_LIST
from graphs.config import RES_DIR
from graphs.gnn import train_model
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import transforms as t

SEEDS = [1, 2, 3, 4, 5]


def benchmark_models(n_layers: int, architecture_list: List[str], gpu_id: int):

    dev_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)

    dataset = PygNodePropPredDataset(
        name="ogbn-arxiv", transform=t.Compose([t.ToUndirected(), t.ToSparseTensor()]), root=DATA_DIR
    )

    data, n_classes, split_idx = dataset[0], dataset.num_classes, dataset.get_idx_split()

    gnn_params = {
        "num_layers": n_layers,
        "in_channels": data.num_features,
        "hidden_channels": 256,
        "dropout": 0.5,
        "out_channels": n_classes,
        "norm": "BatchNorm",
    }
    gat_kwargs = {
        "heads": 3,
        "dropout": 0.05,
    }
    optimizer_params = {"epochs": 500, "lr": 0.01}

    for gnn in architecture_list:

        test_accs = []
        curr_params = copy.deepcopy(gnn_params)

        if gnn == "GAT":
            for k, v in gat_kwargs.items():
                curr_params[k] = v
            curr_params["hidden_channels"] *= gat_kwargs["heads"]
        for seed in SEEDS:
            model = GNN_DICT[gnn](**curr_params)
            benchmark_path = os.path.join(RES_DIR, "test_gnns")
            if not os.path.isdir(benchmark_path):
                os.mkdir(benchmark_path)
            model_path = os.path.join(benchmark_path, gnn)
            if not os.path.isdir(model_path):
                os.mkdir(model_path)
            save_path = os.path.join(model_path, f"model_torch_{gnn}_{n_layers}l_s{seed}.pt")
            train_results, test_acc = train_model(
                model=model,
                data=data,
                split_idx=split_idx,
                device=device,
                seed=seed,
                optimizer_params=optimizer_params,
                save_path=save_path,
                b_test=True,
            )
            print(train_results[-1])

            test_accs.append(test_acc)

            df_train = pd.DataFrame(
                train_results, columns=["Epoch", "Loss", "Training_Accuracy", "Validation_Accuracy", "Test_Accuracy"]
            )
            df_train.to_csv(os.path.join(model_path, f"train_log_torch_{gnn}_{n_layers}l_s{seed}.csv"), index=False)

        print(f"Mean Test Accuracy for torch {gnn} is:")
        print(np.mean(test_accs))


def parse_args():
    """Parses arguments given to script

    Returns:
        dict-like object -- Dict-like object containing all given arguments
    """
    parser = argparse.ArgumentParser()

    # Test parameters
    parser.add_argument(
        "-a", "--architectures", type=str, nargs="+", default=GNN_LIST, choices=GNN_LIST, help="GNN type to benchmark"
    )
    parser.add_argument("-n", "--n_layers", type=int, default=3, help="Number of Layers to benchmark")

    parser.add_argument("--device", type=int, default=0, help="Number of Layers to benchmark")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    benchmark_models(args.n_layers, args.architectures, args.device)
