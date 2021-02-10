import settings
import torch
import os
import numpy as np
from tqdm.auto import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

__all__ = ["Feedforward_model","train_and_predict","predict_ff"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = 'models/bnn'

class Feedforward(nn.Module):

    def __init__(self, n_dim, n_classes):
        super().__init__()
        self.l1 = nn.Linear(n_dim, 32)
        self.l2 = nn.Linear(32,16)
        self.l3 = nn.Linear(16,n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = self.sig(x)
        return x


def train_and_predict(train_dataset, test_dataset, input_dim, output_dim, batch_size, lr, epochs):

    writer = SummaryWriter()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # print(train_loader)


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    model = Feedforward(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for epoch in tqdm(range(epochs), position=0):
        model.train()
        losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.zero_grad()
            output = model(data.float())
            print(target, output)
            loss = loss_fn(output.float(), target.float())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if settings.TENSORBOARD:
            writer.add_scalar('FF/loss/train', np.mean(losses), epoch)
        # To calculate the test loss and write it on tensorboard.
        if (epoch %10 == 0) and (settings.TENSORBOARD):
            predict_ff(model, epoch, test_loader, writer)

    preds, targets = predict_ff(model, epoch, test_loader)


    return (targets, preds)


def predict_ff(net, epoch, test_loader, writer: SummaryWriter = None):

    net.eval()
    loss_fn = nn.BCELoss()

    if isinstance(test_loader.dataset, torch.utils.data.TensorDataset):
        outputs = torch.zeros(test_loader.dataset.tensors[0].shape[0]).to(DEVICE)
        targets = torch.zeros(test_loader.dataset.tensors[0].shape[0])
    else:
        outputs = torch.zeros(test_loader.dataset.__len__()).to(DEVICE)
        targets = torch.zeros(test_loader.dataset.__len__())

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            loc = batch_idx * test_loader.batch_size
            outputs[loc: loc + len(data)] = net(data.float()).reshape(-1)
            targets[loc: loc + len(data)] = target

    loss = loss_fn(outputs, targets)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    if writer is not None:
        writer.add_scalar('FF/loss/test',loss.item(), epoch)

    save_path = os.path.join(MODEL_DIR, f'FF_{epoch}.pth')

    torch.save(net.state_dict(), save_path)

    # return outputs, targets.int()
    return {"predictions": outputs, "targets": targets.int()}


class Feedforward_model:
    base = Feedforward
    args = list()
    kwargs = {}
