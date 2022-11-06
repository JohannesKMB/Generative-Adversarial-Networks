import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
from scipy.optimize import nnls
from tqdm import tqdm
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

DIMENSION = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_sigma_decomp(sigma):
    s, v = torch.linalg.eigh(sigma)
    if torch.any(s <= -1e-6):
        warnings.warn('Covariance matrix has negative eigenvalues: {}'.format(min(s)))
    v = v * torch.sqrt(torch.clip(s, 0, None))[None, :]
    return v

def draw_mixture_dist(mixture_params, num_samples):
    n = len(mixture_params)
    a_mats = torch.stack([get_sigma_decomp(sigma) for w, mu, sigma in mixture_params])

    zs = np.random.choice(n, (num_samples,), p=[w for w, mu, sigma in mixture_params])

    mus = torch.stack([mu for w, mu, sigma in mixture_params])[zs]
    a_mats_samples = a_mats[zs]
    multinomial_samples = torch.randn(num_samples, mus.shape[1], 1, device=device)

    out_samples = mus + torch.matmul(a_mats_samples, multinomial_samples)[..., 0]
    return out_samples

def build_from_2x2(c, mat, m=0, n=0):
    for k, build_list in enumerate(build_candidates_ind(c)):
    
        r_ind = np.random.choice(len(build_list))
        indicees = build_list[r_ind]
        for index in indicees:
            i = index[0]
            j = index[1]
            if k == 0:
                mat[i+m, j+n] = 1
            elif k == 1:
                mat[i+m, j + 2 + n] = 1
            elif k == 2:
                mat[i + 2 + m, j+n] = 1
            else:
                mat[i+2+m, j+2+n] = 1
    return mat

def draw_posterior_matrix(y, mat_dim, num_samples):
    result = []
    for _ in range(num_samples):
        result += [reconstruct(mat_dim, y)]
    return torch.tensor(result, dtype=torch.float32)

def split(c, shape):
    cs = []
    i,j = 0,0
    for _ in range(c.shape[0]):
        for _ in range(c.shape[1]):
            if i%2 == 0 and j%2 == 0:
                try:
                    chat = c[i:i+2, j:j+2]
                    cs += [chat]
                except:
                    print('error', i,j)
            i += 1
            if i >= shape:
                i = 0
        j += 1
        if j >= shape:
                j = 0
    return cs
    
def reconstruct(shape, c):
    splitted = split(c, shape)
    mat = np.zeros((shape, shape))
    m = 0
    i, j = 0,0
    for splitt in splitted:
        mat = build_from_2x2(splitt, mat, i,j)

        i += 4
        m += 1

        if m > 0 and m%int(shape/4) == 0:
            j += 4
            i = 0
            m = 0
            
    return mat

def draw_x_mats(num_samples, shape=4):
    return torch.tensor(np.round(np.random.rand(num_samples, shape, shape)), dtype=torch.float32)

def count(A, stride=2):
    i, j = 0, 0
    M, N = A.shape[0], A.shape[1]
   
    c = torch.zeros(M//stride, N//stride)
    for _ in range(M):
        for _ in range(N):
            if i%2 == 0 and j%2 == 0:
                
                c[i//2, j//2] += A[i, j]
                c[i//2, j//2] += A[i+1, j]
                c[i//2, j//2] += A[i, j+1]
                c[i//2, j//2] += A[i+1, j+1]
                
            i += 1
            if i >= M:
                i = 0
        j += 1
        if j >= N:
                j = 0
    return torch.tensor(c, dtype=torch.float32)

def forward_mat(x):
    m = torch.nn.AvgPool2d(2, stride=(2,2))
    return m(10 * x)

def get_epoch_data_loader_new(num_samples_per_epoch, batch_size, shape):
    x = draw_x_mats(num_samples_per_epoch, shape)
    y = forward_mat(x)
    def epoch_data_loader_new():
        for i in range(0, num_samples_per_epoch, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader_new

def build_candidates(c):
    candidates = []
    cand = []
    for m,row in enumerate(c):
        for n,i in enumerate(row):
            if int(i) == 0 or int(i) == 4:
                cand += [torch.tensor([[i * 0.25, i * 0.25], [i * 0.25, i * 0.25]])]
            elif int(i) == 1:
                cand += [torch.tensor([[1, 0], [0, 0]])]
                cand += [torch.tensor([[0, 1], [0, 0]])]
                cand += [torch.tensor([[0, 0], [1, 0]])]
                cand += [torch.tensor([[0, 0], [0, 1]])]
            elif int(i) == 2:
                cand += [torch.tensor([[1, 1], [0, 0]])]
                cand += [torch.tensor([[0, 0], [1, 1]])]
                cand += [torch.tensor([[1, 0], [1, 0]])]
                cand += [torch.tensor([[0, 1], [0, 1]])]
                cand += [torch.tensor([[1, 0], [0, 1]])]
                cand += [torch.tensor([[0, 1], [1, 0]])]
            else:
                cand += [torch.tensor([[1, 1], [1, 0]])]
                cand += [torch.tensor([[1, 1], [0, 1]])]
                cand += [torch.tensor([[1, 0], [1, 1]])]
                cand += [torch.tensor([[0, 1], [1, 1]])]

            candidates += [cand]
            cand = []
    return candidates

def build_candidates_ind(c):
    candidates = []
    cand = []
    for m,row in enumerate(c):
        for n,i in enumerate(row):
            if int(i) == 0:
                cand += [np.array([])]
            elif int(i) == 2:
                cand += [np.array([(0,0)])]
                cand += [np.array([(0,1)])]
                cand += [np.array([(1,0)])]
                cand += [np.array([(1,1)])]
            elif int(i) == 5:
                cand += [np.array([(0,0), (0,1)])]
                cand += [np.array([(1,0), (1,1)])]
                cand += [np.array([(0,0), (1,0)])]
                cand += [np.array([(0,1), (1,1)])]
                cand += [np.array([(0,0), (1,1)])]
                cand += [np.array([(0,1), (1,0)])]
            elif int(i) == 7:
                cand += [np.array([(0,0), (0,1), (1,0)])]
                cand += [np.array([(0,0), (0,1), (1,1)])]
                cand += [np.array([(0,0), (1,0), (1,1)])]
                cand += [np.array([(0,1), (1,0), (1,1)])]
            else:
                cand += [np.array([(0,0), (0,1), (1,0), (1,1)])]

            candidates += [cand]
            cand = []
    return candidates

def draw_mats_post(y):
    fin = []
    candidates = build_candidates(y)
    for i in candidates[0]:
        for j in candidates[1]:
            for k in candidates[2]:
                for l in candidates[3]:
                    
                    upperb = np.hstack((i, j))
                    lowerb = np.hstack((k, l))
                    f = np.vstack((upperb, lowerb))
                    fin += [f]
    return torch.tensor(fin, dtype=torch.float32)



def get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b):
    x = draw_mixture_dist(mixture_params, num_samples_per_epoch)
    y = forward_pass(x, forward_map)
    y += torch.randn_like(y) * b
    def epoch_data_loader():
        for i in range(0, num_samples_per_epoch, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader


#INN
import torch
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_INN(num_layers, sub_net_size,dimension=5,dimension_condition=5):
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size, sub_net_size), nn.ReLU(),
                             nn.Linear(sub_net_size,  c_out))
    nodes = [InputNode(dimension, name='input')]
    cond = ConditionNode(dimension_condition, name='condition')
    for k in range(num_layers):
        nodes.append(Node(nodes[-1],
                          GLOWCouplingBlock,
                          {'subnet_constructor':subnet_fc, 'clamp':1.4},
                          conditions = cond,
                          name=F'coupling_{k}'))
    nodes.append(OutputNode(nodes[-1], name='output'))

    model = ReversibleGraphNet(nodes + [cond], verbose=False).to(device)
    return model

def train_inn_epoch_mat(optimizer, model, epoch_data_loader, mat_dim):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = x.shape[0]
        x = x.view(-1, mat_dim**2)
        y = y.view(-1, int(mat_dim**2/4))

        loss = 0
        invs, jac_inv = model(x, c = y, rev = True)

        l5 = 0.5 * torch.sum(invs**2, dim=1) - jac_inv
        loss += (torch.sum(l5) / cur_batch_size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss



