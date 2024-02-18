import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, n, J, nhidden, depth = 2):
        super(Model, self).__init__()
        # if y is a vector, suppose it has been added 1 dim such that the batch size = 1.
        # n = len(y[0])
        # n, J = B.size()
        self.fin = nn.Linear(n, nhidden)
        self.linears = nn.ModuleList([nn.Linear(nhidden, nhidden) for i in range(depth)])
        self.fout = nn.Linear(nhidden, J)
        self.activation = nn.GELU()
    def forward(self, y):
        y = self.activation(self.fin(y))
        for i, l in enumerate(self.linears):
            y = self.activation(l(y))
        beta_unsort = self.fout(y)
        beta, indices = torch.sort(beta_unsort)
        return beta

# x: lambda (8xn)
# y: solution of beta (Jxn)
def train_MLP(x, y, xval, yval, gpu_id = 0, eta = 1e-3, nepoch = 100, nhidden = 100, depth = 2):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id != -1 else "cpu"
    x = torch.from_numpy(x).to(device, non_blocking=True)
    xval = torch.from_numpy(xval).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    yval = torch.from_numpy(yval).to(device, non_blocking=True)
    n, J = y.size()
    dim_lam = 8
    model = Model(dim_lam, J, nhidden, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr = eta, amsgrad = True)
    loss_fn = nn.functional.mse_loss
    train_loss = []
    val_loss = []
    for epoch in range(nepoch):
        yhat = model(x)
        loss = loss_fn(yhat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        yval_hat = model(xval)
        with torch.no_grad():
            loss_val = loss_fn(yval_hat, yval)
            val_loss.append(loss_val.item())
    return train_loss, val_loss
