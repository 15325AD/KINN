import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
import sys
sys.path.append("..")
from kan_efficiency import *
import matplotlib.tri as tri

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2024)

# 设备设置
device = 'cpu'
penalty = 1000
delta = 0.0
train_p = 0
nepoch_u0 = 2500
a = 1
elasp = 0.001

# 生成交界面随机点
def interface(Ni):
    re = np.random.rand(Ni)
    theta = 0
    x = np.cos(theta) * re
    y = np.sin(theta) * re
    xi = np.stack([x, y], 1)
    xi = xi.astype(np.float32)
    xi = torch.tensor(xi, requires_grad=True, device=device)
    return xi

# 生成训练数据
def train_data(Nb, Nf):
    def generate_points(square_size, num_points):
        points = []
        for i in range(num_points):
            for j in range(num_points):
                x = -square_size + i * 2 * square_size / (num_points - 1)
                y = -square_size + j * 2 * square_size / (num_points - 1)
                points.append((x, y))
        return np.array(points)

    # 生成点
    square_size = 1
    num_points = 200
    points = generate_points(square_size, num_points)

    # 创建 Delaunay 三角剖分
    triangulation = tri.Triangulation(points[:, 0], points[:, 1])

    # 计算每个三角形的面积
    triangles = points[triangulation.triangles]
    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = 0.5 * (a + b + c)
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))

    # 所有三角形面积的总和
    dom_point = triangles.mean(1)
    total_area = np.sum(areas)
    Xf = torch.tensor(np.hstack((dom_point, areas[:, np.newaxis])))
    print('Total area of the rectangle consisted by the triangle is ' + str(total_area))

    xu = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  np.ones([int(Nb/4),1])]).astype(np.float32)
    xd = np.hstack([np.random.rand(int(Nb/4),1)*2-1,  -np.ones([int(Nb/4),1])]).astype(np.float32)
    xl = np.hstack([-np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1]).astype(np.float32)
    xr = np.hstack([np.ones([int(Nb/4),1]), np.random.rand(int(Nb/4),1)*2-1]).astype(np.float32)
    xcrack = np.hstack([-np.random.rand(int(Nb/4),1), np.zeros([int(Nb/4),1])]).astype(np.float32)
    xc1 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]).astype(np.float32)
    xc2 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]).astype(np.float32)
    xc3 = np.array([[-1., 1.], [1., 1.], [1., -1.], [-1., -1.]]).astype(np.float32)
    Xb = np.concatenate((xu, xd, xl, xr, xcrack, xc1, xc2, xc3))
    Xb = torch.tensor(Xb, device=device)

    Xf1 = Xf[(Xf[:, 1]>0) & (torch.norm(Xf, dim=1)>=delta)]
    Xf2 = Xf[(Xf[:, 1]<0) & (torch.norm(Xf, dim=1)>=delta)]
    Xf1 = torch.tensor(Xf1, requires_grad=True, device=device).float()
    Xf2 = torch.tensor(Xf2, requires_grad=True, device=device).float()
    return Xb, Xf1, Xf2

# 特解网络
class particular(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(particular, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
        self.a1 = torch.Tensor([0.1]).to(device)
        self.n = 1/self.a1.data

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+D_out)))

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)

    def forward(self, x):
        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2))
        y = self.n*self.a1*self.linear4(y3)
        return y

# 齐次解网络
class homo(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(homo, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)
        self.a1 = torch.nn.Parameter(torch.Tensor([0.1])).to(device)
        self.n = 1/self.a1.data

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=np.sqrt(2/(D_in+H)))
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=np.sqrt(2/(H+H)))
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=np.sqrt(2/(H+D_out)))

        torch.nn.init.normal_(self.linear1.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear2.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear3.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear4.bias, mean=0, std=1)
        torch.nn.init.normal_(self.linear5.bias, mean=0, std=1)

    def forward(self, x):
        y1 = torch.tanh(self.n*self.a1*self.linear1(x))
        y2 = torch.tanh(self.n*self.a1*self.linear2(y1))
        y3 = torch.tanh(self.n*self.a1*self.linear3(y2)) + y1
        y4 = torch.tanh(self.n*self.a1*self.linear4(y3)) + y2
        y = self.linear5(y4)
        return y

# 预测函数
def pred(xy):
    pred = torch.zeros((len(xy), 1), device=device)
    pred[(xy[:, 1]>0)] = model_p1(xy[xy[:, 1]>0]) + RBF(xy[xy[:, 1]>0]) * model_h(xy[xy[:, 1]>0])
    pred[xy[:, 1]<0] = model_p2(xy[xy[:, 1]<0]) + RBF(xy[xy[:, 1]<0]) * model_h(xy[xy[:, 1]<0])
    return pred

# 评估函数
def evaluate():
    N_test = 100
    x = np.linspace(-1, 1, N_test).astype(np.float32)
    y = np.linspace(-1, 1, N_test).astype(np.float32)
    x, y = np.meshgrid(x, y)
    xy_test = np.stack((x.flatten(), y.flatten()),1)
    xy_test = torch.from_numpy(xy_test).to(device)

    u_pred = pred(xy_test)
    u_pred = u_pred.data
    u_pred = u_pred.reshape(N_test, N_test).cpu()

    u_exact = np.zeros(x.shape)
    u_exact[y>0] = np.sqrt(np.sqrt(x[y>0]**2+y[y>0]**2))*np.sqrt((1-x[y>0]/np.sqrt(x[y>0]**2+y[y>0]**2))/2)
    u_exact[y<0] = -np.sqrt(np.sqrt(x[y<0]**2+y[y<0]**2))*np.sqrt((1-x[y<0]/np.sqrt(x[y<0]**2+y[y<0]**2))/2)
    u_exact = torch.from_numpy(u_exact)
    error = torch.abs(u_pred - u_exact)
    error_t = torch.norm(error)/torch.norm(u_exact)
    return error_t

# 训练特解网络
if train_p == 1:
    model_p1 = particular(2, 10, 1).to(device)
    model_p2 = particular(2, 10, 1).to(device)
    tol_p = 0.0001
    loss_bn = 100
    epoch_b = 0
    criterion = torch.nn.MSELoss()
    optimp = torch.optim.Adam(params=chain(model_p1.parameters(), model_p2.parameters()), lr=0.0005)
    loss_b_array = []
    loss_b1_array = []
    loss_b2_array = []
    loss_bi_array = []
    loss_bn_array = []

    while loss_bn > tol_p:
        if epoch_b % 10 == 0:
            Xb, Xf1, Xf2 = train_data(256, 4096)
            Xi = interface(1000)
            Xb1 = Xb[Xb[:, 1] >= 0]
            Xb2 = Xb[Xb[:, 1] <= 0]
            target_b1 = torch.sqrt(torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))*torch.sqrt((1-Xb1[:,0]/torch.sqrt(Xb1[:,0]**2+Xb1[:,1]**2))/2)
            target_b1 = target_b1.unsqueeze(1)
            target_b2 = -torch.sqrt(torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))*torch.sqrt((1-Xb2[:,0]/torch.sqrt(Xb2[:,0]**2+Xb2[:,1]**2))/2)
            target_b2 = target_b2.unsqueeze(1)
        epoch_b += 1
        def closure():
            pred_b1 = model_p1(Xb1)
            loss_b1 = criterion(pred_b1, target_b1)
            pred_b2 = model_p2(Xb2)
            loss_b2 = criterion(pred_b2, target_b2)
            loss_b = loss_b1 + loss_b2
            pred_bi1 = model_p1(Xi)
            pred_bi2 = model_p2(Xi)
            loss_bi = criterion(pred_bi1, pred_bi2)
            optimp.zero_grad()
            loss_bn = loss_b + loss_bi
            loss_bn.backward()
            loss_b_array.append(loss_b.data)
            loss_b1_array.append(loss_b1.data)
            loss_b2_array.append(loss_b2.data)
            loss_bi_array.append(loss_bi.data)
            loss_bn_array.append(loss_bn.data)
            if epoch_b % 10 == 0:
                print('Training particular network: epoch %i, loss1 %f, loss2 %f, lossi %f' % (epoch_b, loss_b1.data, loss_b2.data, loss_bi.data))
            return loss_bn
        optimp.step(closure)
        loss_bn = loss_bn_array[-1]
    torch.save(model_p1, './particular1_nn')
    torch.save(model_p2, './particular2_nn')

model_p1 = torch.load('particular1_nn', map_location=torch.device('cpu'))
model_p2 = torch.load('particular2_nn', map_location=torch.device('cpu'))

# RBF 函数
def RBF(x):
    d_total_t = torch.from_numpy(d_total).unsqueeze(1).to(device)
    w_t = torch.from_numpy(w).to(device)
    x_l = x.unsqueeze(0).repeat(len(d_total_t), 1, 1)
    R = torch.norm(d_total_t - x_l, dim=2)
    y = torch.mm(torch.exp(-gama*R.T), w_t)
    return y

# RBF 参数设置
n_d = 10
n_dom = 5
gama = 0.5
ep = np.linspace(-1, 1, n_d).astype(np.float32)
ep1 = np.zeros((n_d, 2)).astype(np.float32)
ep1[:, 0], ep1[:, 1] = ep, 1
ep2 = np.zeros((n_d, 2)).astype(np.float32)
ep2[:, 0], ep2[:, 1] = ep, -1
ep3 = np.zeros((n_d, 2)).astype(np.float32)
ep3[:, 0], ep3[:, 1] = -1, ep
ep4 = np.zeros((n_d, 2)).astype(np.float32)
ep4[:, 0], ep4[:, 1] = 1, ep
ep5 = np.zeros((n_d, 2)).astype(np.float32)
ep5[:, 0], ep5[:, 1] = ep/2-0.5, 0
points_d = np.concatenate((ep1, ep2, ep3, ep4, ep5))
points_d = np.unique(points_d, axis=0)
kdt = KDTree(points_d, metric='euclidean')

domx = np.linspace(-1, 1, n_dom)[1:-1].astype(np.float32)
domy = np.linspace(-1, 1, n_dom)[1:-1].astype(np.float32)
domx, domy = np.meshgrid(domx, domy)
domxy = np.stack((domx.flatten(), domy.flatten()), 1)
domxy = domxy[(domxy[:, 1]!=0)|(domxy[:, 0]>0)]
d_dir, _ = kdt.query(points_d, k=1, return_distance=True)
d_dom, _ = kdt.query(domxy, k=1, return_distance=True)
d_total = np.concatenate((points_d, domxy))
dx = d_total[:, 0][:, np.newaxis] - d_total[:, 0][np.newaxis, :]
dy = d_total[:, 1][:, np.newaxis] - d_total[:, 1][np.newaxis, :]
R = np.sqrt(dx**2+dy**2)
K = np.exp(-gama*R)
b = np.concatenate((d_dir, d_dom))
w = np.dot(np.linalg.inv(K), b).astype(np.float32)

# 测试数据
n_test = 21
domx_t = np.linspace(-1, 1, n_test).astype(np.float32)
domy_t = np.linspace(-1, 1, n_test).astype(np.float32)
domx_t, domy_t = np.meshgrid(domx_t, domy_t)
domxy_t = np.stack((domx_t.flatten(), domy_t.flatten()), 1)
domxy_t = torch.from_numpy(domxy_t).requires_grad_(True).to(device)

dis = RBF(domxy_t)

model_h = KAN([2, 5, 5, 1], base_activation=torch.nn.SiLU, grid_size=10, grid_range=[-1.0, 1.0], spline_order=3)
criterion = torch.nn.MSELoss()
optim_h = torch.optim.Adam([{'params': model_h.parameters()}], lr=0.001)  # 0.001比较好
scheduler = torch.optim.lr_scheduler.MultiStepLR(optim_h, milestones=[500, 3000, 5000], gamma=0.1)
loss_array = []
error_array = []
loss1_array = []
loss2_array = []
lossi_array = []
eigvalues = []
nepoch_u0 = 1000  # 假设 nepoch_u0 是一个整数
start = time.time()

for epoch in range(nepoch_u0):
    if epoch % 100 == 0:
        end = time.time()
        consume_time = end - start
        print('time is %f' % consume_time)
    if epoch % 100 == 0:
        Xb, Xf1_p_a, Xf2_p_a = train_data(100, 4096)

        Xf1 = Xf1_p_a[:, :-1]
        Xf2 = Xf2_p_a[:, :-1]
        Xf1_a = Xf1_p_a[:, -1].unsqueeze(1)
        Xf2_a = Xf2_p_a[:, -1].unsqueeze(1)

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
        u_h1 = model_h(Xf1)
        u_h2 = model_h(Xf2)

        u_p1 = model_p1(Xf1)
        u_p2 = model_p2(Xf2)  # 获得特解
        # 构造可能位移场
        u_pred1 = u_p1 + RBF(Xf1) * u_h1
        u_pred2 = u_p2 + RBF(Xf2) * u_h2

        du1dxy = grad(u_pred1, Xf1, torch.ones(Xf1.size()[0], 1).cpu(), create_graph=True)[0]
        du1dx = du1dxy[:, 0].unsqueeze(1)
        du1dy = du1dxy[:, 1].unsqueeze(1)

        du2dxy = grad(u_pred2, Xf2, torch.ones(Xf2.size()[0], 1).cpu(), retain_graph=True, create_graph=True)[0]
        du2dx = du2dxy[:, 0].unsqueeze(1)
        du2dy = du2dxy[:, 1].unsqueeze(1)

        J1 = torch.sum(0.5 * (du1dx ** 2 + du1dy ** 2) * Xf1_a)

        J2 = torch.sum(0.5 * (du2dx ** 2 + du2dy ** 2) * Xf2_a)

        J = J1 + J2

        # 添加交界面的原函数损失
        u_ii1 = model_p1(Xi) + RBF(Xi) * model_h(Xi)  # 分别得到两个网络的交界面预测
        u_ii2 = model_p2(Xi) + RBF(Xi) * model_h(Xi)
        Ji = criterion(u_ii1, u_ii2)

        loss = J  # + penalty * (Ji + loss_b) # the functional
        error_t = evaluate()
        optim_h.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        error_array.append(error_t.data.cpu())
        loss1_array.append(J1.data.cpu())
        loss2_array.append(J2.data.cpu())
        lossi_array.append(Ji.data.cpu())
        if epoch % 10 == 0:
            print(' epoch : %i, the loss : %f ,  J1 : %f , J2 : %f , Ji : %f, error: %f' % (
            epoch, loss.data, J1.data, J2.data, Ji.data, error_t.data))
        return loss

    optim_h.step(closure)
    scheduler.step()

torch.save(model_h, './model/KINN_DEM_rbf/KINN_DEM_rbf_tri')

np.save('./results/KINN_DEM_rbf/error_tri.npy', np.array(error_array))

# %%
N_test = 100
x = np.linspace(-1, 1, N_test).astype(np.float32)
y = np.linspace(-1, 1, N_test).astype(np.float32)
x, y = np.meshgrid(x, y)
xy_test = np.stack((x.flatten(), y.flatten()), 1)
xy_test = torch.from_numpy(xy_test).cpu()  # 将数据放在CPU上

u_pred = pred(xy_test)
u_pred = u_pred.data
u_pred = u_pred.reshape(N_test, N_test).cpu()

u_exact = np.zeros(x.shape)
u_exact[y > 0] = np.sqrt(np.sqrt(x[y > 0] ** 2 + y[y > 0] ** 2)) * np.sqrt(
    (1 - x[y > 0] / np.sqrt(x[y > 0] ** 2 + y[y > 0] ** 2)) / 2)
u_exact[y < 0] = -np.sqrt(np.sqrt(x[y < 0] ** 2 + y[y < 0] ** 2)) * np.sqrt(
    (1 - x[y < 0] / np.sqrt(x[y < 0] ** 2 + y[y < 0] ** 2)) / 2)
u_exact = torch.from_numpy(u_exact)
error = torch.abs(u_pred - u_exact)  # get the error in every points
error_t = torch.norm(error) / torch.norm(u_exact)  # get the total relative L2 error

# plot the prediction solution
fig = plt.figure(figsize=(20, 20))

plt.subplot(4, 3, 1)
Xb, Xf1, Xf2 = train_data(256, 4096)
Xi1 = interface(1000)
Xi2 = interface(1000)
plt.scatter(Xb.detach().cpu().numpy()[:, 0], Xb.detach().cpu().numpy()[:, 1], c='r', s=0.1)
plt.scatter(Xf1.detach().cpu().numpy()[:, 0], Xf1.detach().cpu().numpy()[:, 1], c='b', s=0.1)
plt.scatter(Xf2.detach().cpu().numpy()[:, 0], Xf2.detach().cpu().numpy()[:, 1], c='g', s=0.1)
plt.scatter(Xi1.detach().cpu().numpy()[:, 0], Xi1.detach().cpu().numpy()[:, 1], c='m', s=0.1)
plt.scatter(Xi2.detach().cpu().numpy()[:, 0], Xi2.detach().cpu().numpy()[:, 1], c='y', s=0.1)

plt.subplot(4, 3, 2)
plt.scatter(d_total[:, 0], d_total[:, 1])
plt.title('the RBF points')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(4, 3, 3)
dis_plot = dis.data.cpu().numpy().reshape(n_test, n_test)
h1 = plt.contourf(domx_t, domy_t, dis_plot, levels=50)
plt.colorbar(h1)
plt.title('the RBF distance function')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(4, 3, 4)
h2 = plt.contourf(x, y, u_pred.detach().numpy(), levels=100, cmap='jet')
plt.title('datadriven prediction')
plt.colorbar(h2)
plt.subplot(4, 3, 5)
h3 = plt.contourf(x, y, u_exact, levels=100, cmap='jet')
plt.title('exact solution')
plt.colorbar(h3)
plt.subplot(4, 3, 6)
h4 = plt.contourf(x, y, error.detach().numpy(), levels=100, cmap='jet')
plt.title('absolute error')
plt.colorbar(h4)
plt.subplot(4, 3, 7)
loss_array = np.array(loss_array)
loss_array = loss_array[loss_array < 50]
plt.yscale('log')
plt.plot(loss_array)
plt.xlabel('the iteration')
plt.ylabel('loss')
plt.title('loss evolution')
plt.subplot(4, 3, 8)
error_array = np.array(error_array)
error_array = error_array[error_array < 1]
plt.yscale('log')
plt.plot(error_array)
plt.xlabel('the iteration')
plt.ylabel('error')
plt.title('relative total error evolution')
plt.subplot(4, 3, 9)
n_inter = 11
interx = torch.linspace(0, 1, n_inter)[1:-1]  # 获得interface的测试点
intery = torch.zeros(n_inter)[1:-1]
inter = torch.stack((interx, intery), 1)
inter = inter.requires_grad_(True).cpu()  # 可以求导并且放入CPU中
pred_inter = (model_p1(inter) + model_p2(inter)) / 2 + \
             RBF(inter) * model_h(inter)
dudxyi = grad(pred_inter, inter, torch.ones(inter.size()[0], 1).cpu())[0]
dudyi = dudxyi[:, 1].unsqueeze(1)  # 获得了交界面奇异应变
dudyi_e = 0.5 / torch.sqrt(torch.norm(inter, dim=1))  # 获得交界面的应变的精确值
inter_strain = np.vstack([interx.cpu(), dudyi.detach().cpu().flatten()])
np.save('./results/KINN_DEM_rbf/interface_tri.npy', inter_strain)  # 将DEM rbf的奇异应变储存起来
plt.plot(interx.cpu(), dudyi_e.detach().cpu(), label='exact', marker='o')
plt.plot(interx.cpu(), dudyi.detach().cpu(), label='predict', marker='*', ls=':')
plt.xlabel('coordinate x')
plt.ylabel('strain e32')
plt.title('cpinn about e32 on interface')
plt.legend()

plt.subplot(4, 3, 10)  # NTK分析
for index, i in enumerate(eigvalues):
    num = 500 * index
    plt.yscale('log')
    plt.plot(i, label='%i' % num)
    plt.title('Spectrum NTK')
plt.legend()
plt.subplot(4, 3, 11)  # 精确解的二维频域图
fre = np.abs(np.fft.fftshift(np.fft.fft2(u_exact.cpu().numpy())))
plt.imshow(fre,  # numpy array generating the image
           # color map used to specify colors
           interpolation='nearest'  # algorithm used to blend square colors; with 'nearest' colors will not be blended
           )
plt.colorbar()
plt.title('the frequency of the exact solution')

plt.subplot(4, 3, 12)  # 预测解的二维频域图
fre_p = np.abs(np.fft.fftshift(np.fft.fft2(u_pred.cpu().numpy())))
plt.imshow(fre_p,  # numpy array generating the image
           # color map used to specify colors
           interpolation='nearest'  # algorithm used to blend square colors; with 'nearest' colors will not be blended
           )
plt.colorbar()
plt.title('the frequency of the pred solution')
plt.suptitle("KINN_DEM_rbf")
plt.savefig('./results/KINN_DEM_rbf/KINN_DEM_rbf_tri.pdf', dpi=300)
plt.show()
print('the relative error is %f' % error_t.data)

# storage the error contouf
h1 = plt.contourf(x, y, error.detach().numpy(), levels=100, cmap='jet')
ax = plt.gca()
ax.set_aspect(1)
plt.colorbar(h1).ax.set_title('abs error')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./results/KINN_DEM_rbf/abs_error_KINN_DEM_rbf_tri.pdf', dpi=300)
plt.show()

sum(p.numel() for p in model_h.parameters())
