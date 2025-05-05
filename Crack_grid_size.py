import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
from itertools import chain
import sys

sys.path.append("..")
from kan_efficiency import *


def setup_seed(seed):
    # random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2024)

# Device configuration - using CPU
device = torch.device('cpu')

penalty = 1
delta = 0.0
train_p = 0
nepoch_u0 = 2500
a = 1

b1 = 1
b2 = 1
b3 = 50
b4 = 50
b5 = 10


def interface(Ni):
    '''
    生成交界面的随机点
    '''
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = xi.astype(np.float32)
    xi = torch.tensor(xi, requires_grad=True, device=device)
    return xi


def train_data(Nb, Nf):
    '''
    生成强制边界点，四周以及裂纹处
    生成上下的内部点
    '''
    xu = np.hstack([np.random.rand(int(Nb / 4), 1) * 2 - 1, np.ones([int(Nb / 4), 1])]).astype(np.float32)
    xd = np.hstack([np.random.rand(int(Nb / 4), 1) * 2 - 1, -np.ones([int(Nb / 4), 1])]).astype(np.float32)
    xl = np.hstack([-np.ones([int(Nb / 4), 1]), np.random.rand(int(Nb / 4), 1) * 2 - 1]).astype(np.float32)
    xr = np.hstack([np.ones([int(Nb / 4), 1]), np.random.rand(int(Nb / 4), 1) * 2 - 1]).astype(np.float32)
    xcrack = np.hstack([-np.random.rand(int(Nb / 4), 1), np.zeros([int(Nb / 4), 1])]).astype(np.float32)

    # 随机撒点时候，边界没有角点，这里要加上角点
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]], dtype=np.float32)
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]], dtype=np.float32)
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]], dtype=np.float32)

    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3))
    Xb = torch.tensor(Xb, device=device)

    Xf = torch.rand(Nf, 2) * 2 - 1

    Xf1 = Xf[(Xf[:, 1] > 0) & (torch.norm(Xf, dim=1) >= delta)]  # 上区域点
    Xf2 = Xf[(Xf[:, 1] < 0) & (torch.norm(Xf, dim=1) >= delta)]  # 下区域点

    Xf1 = torch.tensor(Xf1, requires_grad=True, device=device)
    Xf2 = torch.tensor(Xf2, requires_grad=True, device=device)

    return Xb, Xf1, Xf2


class particular(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(particular, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)

        self.a1 = torch.Tensor([0.1]).to(device)
        self.a2 = torch.Tensor([0.1])
        self.a3 = torch.Tensor([0.1])
        self.n = 1 / self.a1.data.to(device)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2 / (D_in + H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2 / (H + H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2 / (H + H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2 / (H + D_out)))

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)

    def forward(self, x):
        y1 = torch.tanh(self.n * self.a1 * self.linear1(x))
        y2 = torch.tanh(self.n * self.a1 * self.linear2(y1))
        y3 = torch.tanh(self.n * self.a1 * self.linear3(y2))
        y = self.n * self.a1 * self.linear4(y3)
        return y


def pred(xy, model1, model2):
    pred = torch.zeros((len(xy), 1), device=device)
    pred[(xy[:, 1] > 0)] = model1(xy[xy[:, 1] > 0])
    pred[xy[:, 1] < 0] = model2(xy[xy[:, 1] < 0])
    return pred


def evaluate(model1, model2):
    N_test = 100
    x = np.linspace(-1, 1, N_test).astype(np.float32)
    y = np.linspace(-1, 1, N_test).astype(np.float32)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()), 1)
    xy_test = torch.from_numpy(xy_test).to(device)

    u_pred = pred(xy_test, model1, model2)
    u_pred = u_pred.data
    u_pred = u_pred.reshape(N_test, N_test).cpu()

    u_exact = np.zeros(x.shape)
    u_exact[y > 0] = np.sqrt(np.sqrt(x[y > 0] ** 2 + y[y > 0] ** 2)) * np.sqrt(
        (1 - x[y > 0] / np.sqrt(x[y > 0] ** 2 + y[y > 0] ** 2)) / 2)
    u_exact[y < 0] = -np.sqrt(np.sqrt(x[y < 0] ** 2 + y[y < 0] ** 2)) * np.sqrt(
        (1 - x[y < 0] / np.sqrt(x[y < 0] ** 2 + y[y < 0] ** 2)) / 2)

    u_exact = u_exact.reshape(x.shape)
    u_exact = torch.from_numpy(u_exact)
    error = torch.abs(u_pred - u_exact)
    error_t = torch.norm(error) / torch.norm(u_exact)
    return error_t


def run_scale_law(layers, grid_size, epoch_num=2500):
    model_p1 = KAN(layers, base_activation=torch.nn.SiLU, grid_size=grid_size, grid_range=[-1.0, 1.0],
                   spline_order=3).to(device)
    model_p2 = KAN(layers, base_activation=torch.nn.SiLU, grid_size=grid_size, grid_range=[-1.0, 1.0],
                   spline_order=3).to(device)

    criterion = torch.nn.MSELoss()
    optim_h = torch.optim.Adam(params=chain(model_p1.parameters(), model_p2.parameters()), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[3000, 5000, 5000], gamma=0.1)

    loss_array = []
    error_array = []
    start = time.time()

    for epoch in range(epoch_num):
        if epoch % 1000 == 0:
            end = time.time()
            consume_time = end - start
            print('time is %f' % consume_time)

        if epoch % 100 == 0:
            Xb, Xf1, Xf2 = train_data(256, 4096)
            Xi = interface(1000)

            Xb1 = Xb[Xb[:, 1] >= 0]  # 上边界点
            Xb2 = Xb[Xb[:, 1] <= 0]  # 下边界点
            target_b1 = torch.sqrt(torch.sqrt(Xb1[:, 0] ** 2 + Xb1[:, 1] ** 2)) * torch.sqrt(
                (1 - Xb1[:, 0] / torch.sqrt(Xb1[:, 0] ** 2 + Xb1[:, 1] ** 2)) / 2)
            target_b1 = target_b1.unsqueeze(1)
            target_b2 = -torch.sqrt(torch.sqrt(Xb2[:, 0] ** 2 + Xb2[:, 1] ** 2)) * torch.sqrt(
                (1 - Xb2[:, 0] / torch.sqrt(Xb2[:, 0] ** 2 + Xb2[:, 1] ** 2)) / 2)
            target_b2 = target_b2.unsqueeze(1)

        def closure():
            global b1, b2, b3, b4, b5

            # Forward pass
            u_pred1 = model_p1(Xf1)
            u_pred2 = model_p2(Xf2)

            # First derivatives
            du1dxy = grad(u_pred1.sum(), Xf1, create_graph=True)[0]
            du1dx = du1dxy[:, 0].unsqueeze(1)
            du1dy = du1dxy[:, 1].unsqueeze(1)

            du2dxy = grad(u_pred2.sum(), Xf2, create_graph=True)[0]
            du2dx = du2dxy[:, 0].unsqueeze(1)
            du2dy = du2dxy[:, 1].unsqueeze(1)

            # Second derivatives
            du1dxx = grad(du1dx.sum(), Xf1, create_graph=True)[0][:, 0].unsqueeze(1)
            du1dyy = grad(du1dy.sum(), Xf1, create_graph=True)[0][:, 1].unsqueeze(1)

            du2dxx = grad(du2dx.sum(), Xf2, create_graph=True)[0][:, 0].unsqueeze(1)
            du2dyy = grad(du2dy.sum(), Xf2, create_graph=True)[0][:, 1].unsqueeze(1)

            # PDE losses
            J1 = torch.mean((du1dxx + du1dyy) ** 2)
            J2 = torch.mean((du2dxx + du2dyy) ** 2)

            # Boundary losses
            pred_b1 = model_p1(Xb1)
            loss_b1 = criterion(pred_b1, target_b1)
            pred_b2 = model_p2(Xb2)
            loss_b2 = criterion(pred_b2, target_b2)

            # Interface losses
            pred_bi1 = model_p1(Xi)
            pred_bi2 = model_p2(Xi)
            loss_bi = criterion(pred_bi1, pred_bi2)

            du1dxyii = grad(pred_bi1.sum(), Xi, create_graph=True)[0][:, 1].unsqueeze(1)
            du2dxyii = grad(pred_bi2.sum(), Xi, create_graph=True)[0][:, 1].unsqueeze(1)
            loss_bdi = criterion(du1dxyii, du2dxyii)

            # Total loss
            loss = b1 * J1 + b2 * J2 + b3 * loss_b1 + b4 * loss_b2 + b5 * (loss_bi + loss_bdi)

            error_t = evaluate(model_p1, model_p2)
            optim_h.zero_grad()
            loss.backward()

            loss_array.append(loss.item())
            error_array.append(error_t.item())

            if epoch % 10 == 0:
                print('epoch: %i, loss: %f, J1: %f, J2: %f, Jb: %f, Ji: %f, error: %f' %
                      (epoch, loss.item(), J1.item(), J2.item(), (loss_b1 + loss_b2).item(),
                       (loss_bi + loss_bdi).item(), error_t.item()))

            return loss

        optim_h.step(closure)
        scheduler.step()

    num_para = sum(p.numel() for p in model_p1.parameters())
    return num_para, error_array[-1]


if __name__ == "__main__":
    para_list = []
    error_list = []
    layers_list = [[2, 5, 1], [2, 5, 5, 1], [2, 5, 5, 5, 1]]
    grid_size_list = [3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 27, 30]

    dict_scale_law = {}
    for layer in layers_list:
        for grid_size in grid_size_list:
            print(f"Running with layers {layer} and grid_size {grid_size}")
            para_e, error_e = run_scale_law(layer, grid_size, epoch_num=3500)
            para_list.append(para_e)
            error_list.append(error_e)

    dict_scale_law = {
        'layers_list': layers_list,
        'grid_size_list': grid_size_list,
        'parameters': para_list,
        'errors': error_list
    }

    np.save('./results/Crack_grid_size.npy', dict_scale_law)
    loaded_data = np.load('./results/Crack_grid_size.npy', allow_pickle=True).item()
    print(loaded_data)