import numpy as np
import torch
import torch.nn.functional as F
from data.multi_mnist import MultiMNIST
from loss_weight import UncertainLossWeighter
from net.lenet import MultiLeNetO
from net.lenet import MultiLeNetR
from torchvision import transforms

# from utils import create_logger

# ------------------ CHANGE THE CONFIGURATION -------------
PATH = "./dataset"
LR = 0.0005
BATCH_SIZE = 256
NUM_EPOCHS = 128
TASKS = ["R", "L"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE, flush=True)

# ---------------------------------------------------------


accuracy = lambda logits, gt: ((logits.argmax(dim=-1) == gt).float()).mean()
to_dev = lambda inp, dev: [x.to(dev) for x in inp]
# logger = create_logger('Main')

global_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dst = MultiMNIST(PATH, train=True, download=True, transform=global_transformer, multi=True)
train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dst = MultiMNIST(PATH, train=False, download=True, transform=global_transformer, multi=True)
val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=1)
nets = {"rep": MultiLeNetR().to(DEVICE), "L": MultiLeNetO().to(DEVICE), "R": MultiLeNetO().to(DEVICE)}

num_tasks = 2
# loss_weighter = None
loss_weighter = UncertainLossWeighter(num_tasks).to(DEVICE)

print("Using loss_weighter:", loss_weighter, flush=True)

if loss_weighter is not None:
    params = [p for v in nets.values() for p in list(v.parameters())] + list(loss_weighter.parameters())
else:
    params = [p for v in nets.values() for p in list(v.parameters())]

optimizer = torch.optim.Adam(params, lr=LR)

grad_optimizer = None
# grad_optimizer = pc_grad(optimizer)

print("Training starts", flush=True)
total_steps = 0
for ep in range(NUM_EPOCHS):
    print("Training epoch {}/{} ...".format(ep + 1, NUM_EPOCHS), flush=True)
    for net in nets.values():
        net.train()
    for batch in train_loader:
        mask = None
        optimizer.zero_grad()
        img, label_l, label_r = to_dev(batch, DEVICE)
        rep, mask = nets["rep"](img, mask)
        out_l, mask_l = nets["L"](rep, None)
        out_r, mask_r = nets["R"](rep, None)

        losses = [F.nll_loss(out_l, label_l), F.nll_loss(out_r, label_r)]
        if loss_weighter is not None:
            losses = loss_weighter(losses)
        if grad_optimizer is not None:
            grad_optimizer.pc_backward(losses)
            grad_optimizer.step()
        else:
            sum(losses).backward()
            optimizer.step()
        total_steps += 1
        if (total_steps % 100) == 0:
            print("Step #{:.0f}".format(total_steps), flush=True)

    print("Evaluating ...", flush=True)
    losses, acc = [], []
    for net in nets.values():
        net.eval()
    for batch in val_loader:
        img, label_l, label_r = to_dev(batch, DEVICE)
        mask = None
        rep, mask = nets["rep"](img, mask)
        out_l, mask_l = nets["L"](rep, None)
        out_r, mask_r = nets["R"](rep, None)

        losses.append([F.nll_loss(out_l, label_l).item(), F.nll_loss(out_r, label_r).item()])
        acc.append([accuracy(out_l, label_l).item(), accuracy(out_r, label_r).item()])
    losses, acc = np.array(losses), np.array(acc)
    print(
        "Epoches {}/{}: loss (left, right) = {:5.4f}, {:5.4f}".format(
            ep + 1, NUM_EPOCHS, losses[:, 0].mean(), losses[:, 1].mean()
        ),
        flush=True,
    )
    print(
        "Epoches {}/{}: accuracy (left, right) = {:5.3f}, {:5.3f}".format(
            ep + 1, NUM_EPOCHS, acc[:, 0].mean(), acc[:, 1].mean()
        ),
        flush=True,
    )
    # logger.info('epoches {}/{}: loss (left, right) = {:5.4f}, {:5.4f}'.format(
    #     ep, NUM_EPOCHS, losses[:,0].mean(), losses[:,1].mean()))
    # logger.info(
    #     'epoches {}/{}: accuracy (left, right) = {:5.3f}, {:5.3f}'.format(
    #         ep, NUM_EPOCHS, acc[:,0].mean(), acc[:,1].mean()))
