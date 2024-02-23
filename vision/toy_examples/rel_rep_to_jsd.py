from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.metrics.jensen_shannon_distance import jensen_shannon_distance
from vision.util import data_structs as ds
from vision.util import find_architectures as fa
from vision.util.default_params import get_default_arch_params
from vision.util.find_datamodules import get_datamodule


def collect_models_to_compare(architecture: ds.BaseArchitecture, dataset: ds.Dataset):
    """Collects all models that should be compared."""

    root_path = Path(
        f"/mnt/cluster-checkpoint-all/t006d/results/knowledge_extension_iclr24/FIRST_MODELS__{dataset.value}__{architecture.value}"
    )
    assert root_path.exists()

    ckpt_paths = []
    for p in root_path.iterdir():
        ckpt = p / "checkpoints" / "final.ckpt"
        if ckpt.exists():
            ckpt_paths.append(ckpt)
    return ckpt_paths


def extract_representations(
    model: AbsActiExtrArch,
    dataloader: torch.utils.data.DataLoader,
    rel_reps: dict[str, torch.Tensor] = None,
    meta_info: bool = True,
    remain_spatial: bool = False,
) -> dict[str, torch.Tensor]:
    """Extracts the anchor representations from the model."""
    reps: dict[str, list[torch.Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if rel_reps is None:
        for cnt, hook in enumerate(model.hooks):
            reps[str(cnt)] = []
            handles.append(model.register_parallel_rep_hooks(hook, reps[str(cnt)], remain_spatial))
    else:
        for cnt, hook in enumerate(model.hooks):
            reps[str(cnt)] = []
            handles.append(model.register_relative_rep_hooks(hook, rel_reps[str(cnt)], reps[str(cnt)]))

    logits = []
    probs = []
    labels = []
    pred_cls = []
    model.eval()
    model.cuda()
    with torch.no_grad():
        for cnt, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if cnt > 50:
                continue
            im, lbl = batch
            y_logit = model(im.cuda())
            y_probs = torch.softmax(y_logit, dim=1)
            y_hat = torch.argmax(y_probs, dim=1)
            logits.append(y_logit.cpu())
            probs.append(y_probs.cpu())
            labels.append(lbl.cpu())
            pred_cls.append(y_hat.cpu())

    for handle in handles:
        handle.remove()

    for cnt, rep in reps.items():
        tmp_reps = torch.cat(rep, dim=0)
        if remain_spatial:
            reps[cnt] = tmp_reps
        else:
            reps[cnt] = torch.reshape(tmp_reps, (tmp_reps.shape[0], -1))  # Flatten the into Samples x Features
    out_reps = {"reps": reps}
    if meta_info:
        out_reps["logits"] = torch.cat(logits, dim=0)
        out_reps["probs"] = torch.cat(probs, dim=0)
        out_reps["y_hat"] = torch.cat(pred_cls, dim=0)
        out_reps["gt"] = torch.cat(labels, dim=0)
    return out_reps


def compare_rel_reps(rel_reps_a: dict[int, torch.Tensor], rel_reps_b: dict[int, torch.Tensor]):
    jsd = jensen_shannon_distance(rel_reps_a["probs"], rel_reps_b["probs"], aggregated=False).cpu()
    jsd_std_normal = ((jsd - jsd.mean()) / jsd.std(0)).numpy()
    cos_sim = {}
    cos_sim_std_normal = {}
    for k, reps_a in rel_reps_a["reps"].items():
        reps_b = rel_reps_b["reps"][k]
        c_sim = F.cosine_similarity(reps_a, reps_b, dim=-1).cpu()
        cos_sim[k] = c_sim.numpy()
        cos_sim_std_normal[k] = ((c_sim - c_sim.mean()) / c_sim.std()).numpy()
    res = {
        "jsd": jsd,
        "jsd_std_normal": jsd_std_normal,
        "cos_sim": cos_sim,
        "cos_sim_std_normal": cos_sim_std_normal,
    }
    return res


def compare_models(
    model_a: AbsActiExtrArch, model_b: AbsActiExtrArch, val_dataloader, test_dataloader, anchor_dataloader
):
    """Register hooks for each architecture"""
    with torch.no_grad():
        anchor_reps_a = extract_representations(model_a, anchor_dataloader, rel_reps=None, meta_info=False)
        rel_reps_a = extract_representations(model_a, val_dataloader, rel_reps=anchor_reps_a, meta_info=True)

        anchor_reps_b = extract_representations(model_b, anchor_dataloader, rel_reps=None, meta_info=False)
        rel_reps_b = extract_representations(model_b, val_dataloader, rel_reps=anchor_reps_b["reps"], meta_info=True)

        val_res = compare_rel_reps(rel_reps_a, rel_reps_b)

        del rel_reps_a, rel_reps_b

        rel_reps_a = extract_representations(model_a, test_dataloader, rel_reps=anchor_reps_a["reps"], meta_info=True)
        rel_reps_b = extract_representations(model_b, test_dataloader, rel_reps=anchor_reps_b["reps"], meta_info=True)
        test_res = compare_rel_reps(rel_reps_a, rel_reps_b)
        return val_res, test_res


def load_model(ckpt_path: Path, architecture: ds.BaseArchitecture, dataset: ds.Dataset):
    """Loads a model from a checkpoint."""
    architecture_kwargs = get_default_arch_params(dataset)
    model = fa.get_base_arch(architecture)(**architecture_kwargs)
    model.load_state_dict(torch.load(ckpt_path))
    return model


def compare_all_models(ckpt_paths: list[Path], architecture: ds.BaseArchitecture, dataset: ds.Dataset) -> list[dict]:
    """Compare all models."""
    datamodule = get_datamodule(dataset)
    anchor_datamodule = datamodule.anchor_dataloader()
    val_datamodule = datamodule.val_dataloader(0, ds.Augmentation.VAL, batch_size=100)
    test_datamodule = datamodule.test_dataloader(batch_size=100)

    all_results = []
    for combi in combinations(ckpt_paths, 2):
        ckpt_a, ckpt_b = combi
        model_a = load_model(ckpt_a, architecture, dataset)
        model_b = load_model(ckpt_b, architecture, dataset)
        all_results.append(compare_models(model_a, model_b, val_datamodule, test_datamodule, anchor_datamodule))
    return all_results


def main():
    architecture = ds.BaseArchitecture.RESNET18
    dataset = ds.Dataset.CIFAR100
    ckpt_paths = collect_models_to_compare(architecture, dataset)
    results = compare_all_models(ckpt_paths, architecture, dataset)
    print(results)

    # Now print the similarities nicely


if __name__ == "__main__":
    main()
