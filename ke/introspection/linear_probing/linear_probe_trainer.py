import torch
from ke.arch import abstract_acti_extr
from ke.util import data_structs as ds
from ke.util.find_datamodules import get_datamodule
from torch import nn
from torch.nn import functional as F


def get_flat_layer_output(container: list, post_pool_size: int = 1):
    def hook(model, inp, output):
        """
        Attaches a forward hook that takes the output of a layer,
        checks how high the spatial extent is and only saves as many values
        of the representations as passed in wrapper `wanted_spatial`.

        ATTENTION: This procedure removes location information, making intra-layer comparisons
        based off pooling or something like it impossible!
        """
        # self.activations.append(output.detach().cpu().numpy())
        # B x C x H x W
        reduced_output = F.adaptive_avg_pool2d(output, output_size=post_pool_size)  # B x C x postpool x postpool
        flat_output = torch.reshape(reduced_output, (reduced_output.shape[0], -1))  # B x C * (postpool**2)

        container.append(flat_output)

    return hook


class LinearProbeTrainer(nn.Module):
    def __init__(self, module: abstract_acti_extr.AbsActiExtrArch, training_info: ds.FirstModelInfo):
        """
        Trainer learning linear probes for each hook of the model.
        These can be learned to compare models.
        """

        dataloader = get_datamodule(ds.Dataset(training_info.dataset))
        # To select the Augmentations
        self.train_info = training_info
        self.out_cls = dataloader.n_classes
        self.dataloader = dataloader.val_dataloader(training_info.split, ds.Augmentation.VAL)
        self.module: abstract_acti_extr.AbsActiExtrArch = module
        self.containers: dict[str, list] = {}
        self.handles = []
        self.linear_probes: dict[str, nn.Module] = {}

    def register_hooks(self):
        h: ds.Hook
        for h in self.module.hooks:
            wanted_module = self.module.get_wanted_module(h)
            self.containers[h.name] = []
            handle = wanted_module.register_forward_hook(get_flat_layer_output(self.containers[h.name]))
            self.handles.append(handle)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def create_linear_probes(self, hooks_of_interest):
        """
        Creates all prediction layers used or the linear probes later on.
        """

        for h in hooks_of_interest:
            self.linear_probes[h.name] = nn.Linear(in_features=h.n_channels, out_features=self.out_cls)

    def grab_activations(self) -> dict[str, torch.Tensor]:
        """
        Grabs all activations (averaged over channels, so should easily fit to memory for not too large Datasets)
        """
        with torch.no_grad():
            for batch in self.dataloader:
                x, y = batch
                self.model(x)
            activations: dict[str, torch.Tensor] = {}
            for k, v in self.containers.items():
                activations[k] = torch.stack(v, dim=0)
        return activations

    def fit_all_layers(self):
        pass

    @staticmethod
    def fit_intermediate_layer(activations: dict, linear_probes: dict[str, nn.Module], gt: torch.Tensor):
        """Trains the linear probe using the LBFGS optimizer"""
        for name in activations.keys():
            acti = activations[name]
            lin_probe = linear_probes[name]

            optim = torch.optim.LBFGS(params=lin_probe.parameters(), lr=1, max_iter=100)  # Tune this.
            ce_loss = nn.CrossEntropyLoss()

            # Define the closure for L-BFGS optimization
            def closure() -> float:
                optim.zero_grad()
                predictions = lin_probe(acti)
                loss = ce_loss(predictions, gt)
                loss.backward()
                return loss

            # Initialize the L-BFGS optimizer

            optim.step(closure)
