import torch
from tqdm import tqdm
from vision.arch.abstract_acti_extr import AbsActiExtrArch
from vision.util import data_structs as ds


def extract_single_layer_representations(
    layer_id: int,
    model: AbsActiExtrArch,
    dataloader: torch.utils.data.DataLoader,
    rel_reps: dict[str, torch.Tensor] = None,
    meta_info: bool = True,
    remain_spatial: bool = False,
) -> dict[str, torch.Tensor]:
    """Extracts the anchor representations from the model."""
    reps: list[torch.Tensor] = []
    handle: torch.utils.hooks.RemovableHandle
    if rel_reps is None:
        handle = model.register_parallel_rep_hooks(model.hooks[layer_id], reps, remain_spatial)
    else:
        raise NotImplementedError("Relative representations not used here.")

    logits = []
    probs = []
    labels = []
    pred_cls = []
    model.eval()
    model.cuda()
    with torch.no_grad():
        for cnt, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting Representations"):
            if cnt > 50:
                continue
            im, lbl = batch[0], batch[1]
            y_logit = model(im.cuda())
            y_probs = torch.softmax(y_logit, dim=1)
            y_hat = torch.argmax(y_probs, dim=1)
            logits.append(y_logit.cpu())
            probs.append(y_probs.cpu())
            labels.append(lbl.cpu())
            pred_cls.append(y_hat.cpu())

    handle.remove()

    reps = torch.cat(reps, dim=0)
    if not remain_spatial:
        reps = torch.reshape(reps, (reps.shape[0], -1))  # Flatten the into Samples x Features
    out = {"reps": reps}
    if meta_info:
        out["logits"] = torch.cat(logits, dim=0)
        out["probs"] = torch.cat(probs, dim=0)
        out["y_hat"] = torch.cat(pred_cls, dim=0)
        out["gt"] = torch.cat(labels, dim=0)
    return out


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