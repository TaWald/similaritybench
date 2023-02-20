import numpy as np
import torch
import torch.nn.functional as F
from data.multi_mnist import MultiMNIST
from net.lenet import MultiLeNetO
from net.lenet import MultiLeNetR
from pcgrad_amp import PCGradAMP
from torchvision import transforms

# ------------------ CHANGE THE CONFIGURATION -------------
PATH = "./dataset"
LR = 0.0005
BATCH_SIZE = 64
ACCUM_STEPS = 4
NUM_EPOCHS = 128
TASKS = ["R", "L"]
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Using device:", DEVICE, flush=True)
# ---------------------------------------------------------

accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
to_dev = lambda inp, dev: [x.to(dev) for x in inp]

global_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dst = MultiMNIST(PATH, train=True, download=True, transform=global_transformer, multi=True)
train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dst = MultiMNIST(PATH, train=False, download=True, transform=global_transformer, multi=True)
val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=1)
nets = {"rep": MultiLeNetR().to(DEVICE), "L": MultiLeNetO().to(DEVICE), "R": MultiLeNetO().to(DEVICE)}

use_amp = True
# scaler = None
scaler = torch.cuda.amp.GradScaler()

num_tasks = 2

loss_weighter = None
# loss_weighter = UncertainLossWeighter(num_tasks).to(DEVICE)

print("Using loss_weighter:", loss_weighter, flush=True)

if loss_weighter is not None:
    params = [p for v in nets.values() for p in list(v.parameters())] + list(loss_weighter.parameters())
else:
    params = [p for v in nets.values() for p in list(v.parameters())]

optimizer = torch.optim.Adam(params, lr=LR)

lr_scheduler = None

# grad_optimizer = None
grad_optimizer = PCGradAMP(num_tasks, optimizer, scaler=scaler, reduction="sum", cpu_offload=False)
# grad_optimizer = GradVacAMP(num_tasks, optimizer, DEVICE, scaler = scaler, beta = 1e-2, reduction='sum', cpu_offload = False)

print("using grad_optimizer:", grad_optimizer, flush=True)

print("Training starts", flush=True)
total_steps = 0
for ep in range(NUM_EPOCHS):
    print("Training epoch {}/{} ...".format(ep + 1, NUM_EPOCHS), flush=True)
    losses1 = 0.0
    losses2 = 0.0
    count = 0.0
    for net in nets.values():
        net.train()
    for batch in train_loader:
        mask = None
        optimizer.zero_grad()
        img, label_l, label_r = to_dev(batch, DEVICE)
        with torch.cuda.amp.autocast(enabled=use_amp):
            rep, mask = nets["rep"](img, mask)
            out_l, mask_l = nets["L"](rep, None)
            out_r, mask_r = nets["R"](rep, None)
            losses = [F.nll_loss(out_l, label_l) / ACCUM_STEPS, F.nll_loss(out_r, label_r) / ACCUM_STEPS]
            if loss_weighter is not None:
                losses = loss_weighter(losses)

            losses1 += losses[0].item() * ACCUM_STEPS
            losses2 += losses[1].item() * ACCUM_STEPS
            count += 1.0

        if grad_optimizer is not None:
            grad_optimizer.backward(losses)
        else:
            if scaler is not None:
                scaler.scale(sum(losses)).backward()
            else:
                sum(losses).backward()

        total_steps += 1
        if total_steps % ACCUM_STEPS == 0:
            if lr_scheduler is not None:
                lr_scheduler.step()
            if grad_optimizer is not None:
                grad_optimizer.step()
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            if (total_steps % (ACCUM_STEPS * 100)) == 0:
                print("Step #{:.0f}".format(total_steps / ACCUM_STEPS), flush=True)

    print("Epoch #{:.0f} LOSS: {:.5f},{:.5f}".format(ep + 1, losses1 / count, losses2 / count), flush=True)

    print("Evaluating ...", flush=True)
    losses, acc = [], []
    for net in nets.values():
        net.eval()
    for batch in val_loader:
        img, label_l, label_r = to_dev(batch, DEVICE)
        mask = None
        with torch.cuda.amp.autocast(enabled=use_amp):
            rep, mask = nets["rep"](img, mask)
            out_l, mask_l = nets["L"](rep, None)
            out_r, mask_r = nets["R"](rep, None)

            losses.append([F.nll_loss(out_l, label_l).item(), F.nll_loss(out_r, label_r).item()])
            acc.append([accuracy(out_l, label_l).item(), accuracy(out_r, label_r).item()])
    losses, acc = np.array(losses), np.array(acc)
    print(
        "Epoches {}/{}: loss (left, right) = {:5.5f},{:5.5f}".format(
            ep + 1, NUM_EPOCHS, losses[:, 0].mean(), losses[:, 1].mean()
        ),
        flush=True,
    )
    print(
        "Epoches {}/{}: accuracy (left, right) = {:5.5f},{:5.5f}".format(
            ep + 1, NUM_EPOCHS, acc[:, 0].mean(), acc[:, 1].mean()
        ),
        flush=True,
    )
