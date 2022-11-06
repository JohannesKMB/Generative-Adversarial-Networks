import numpy as np
import matplotlib.pyplot as plt
import warnings
import torch
from scipy.optimize import nnls
from tqdm import tqdm
from torch import nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet, ConditionNode
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom
from sklearn.datasets import fetch_openml
from sklearn import preprocessing


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
###########################################################################################

def draw_x_y(num_samples):
    x = []
    y = []
    for _ in range(num_samples):
        k = np.random.choice(8)
        x += [np.random.multivariate_normal([10 * np.cos((k-1)*2 * np.pi/ 8) / np.sin(np.pi/8),
                                             10 * np.sin((k-1)*2 * np.pi/ 8)/ np.sin(np.pi/8)],
                                            [[0.4, 0], [0, 0.4]], size=1)]
        if k in range(4):
            y += [[1,0,0,0]]
        elif k in range(4,6):
            y += [[0,1,0,0]]
        elif k in range(6,7):
            y += [[0,0,1,0]]
        else:
            y += [[0,0,0,1]]
    return np.array(x).reshape(-1,2), np.array(y)
def convert(y):
    col = []
    for i,c in enumerate(y):
        if c[0] == 1:
            col += [(1,0,0)]
        elif c[1] == 1:
            col += [(0,1,0)]
        elif c[2] == 1:
            col += [(0,0,1)]
        else:
            col += [(1,0,1)]
    return np.array(col)
def get_epoch_data_loader_mnist(num_samples_per_epoch, batch_size):
    mnist = fetch_openml('mnist_784')
    x,y = mnist.data, mnist.target
    x = np.array(x, dtype=np.float32)
    x = preprocessing.normalize(x, norm='max')
    y = np.array(y, dtype=np.float32)
    y = np.array([np.eye(1, 10, int(n)) for n in y])
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    def epoch_data_loader_new():
        for i in range(0, num_samples_per_epoch, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader_new
def draw_posterior(y, num_samples):
    ind = list(y).index(1)
    if ind == 1:
        k = np.random.choice(4)
        
    elif ind == 2:
        k = np.random.choice((4,5))
    elif ind == 3:
        k = 6
    else:
        k = 7
                                        
    x = [np.random.multivariate_normal([np.cos((k-1)*2 * np.pi/ 8) / np.sin(np.pi/8),
                                 np.sin((k-1)*2 * np.pi/ 8)/ np.sin(np.pi/8)],
                                        [[0.04, 0], [0, 0.04]], size=num_samples)]
    return np.array(x).reshape(-1,2)

'''
def draw_mixture_dist(mixture_params, num_samples):
    n = len(mixture_params)
    sigmas=torch.stack([torch.sqrt(sigma) for w,mu,sigma in mixture_params])
    probs=np.array([w for w, mu, sigma in mixture_params])
    zs = np.random.choice(n, (num_samples,), p=probs/probs.sum())
    mus = torch.stack([mu for w, mu, sigma in mixture_params])[zs]
    sigmas_samples = sigmas[zs]
    multinomial_samples = torch.randn(num_samples, mus.shape[1], device=device)
    #print(mus.shape)
    #print((multinomial_samples).shape)
    #print((sigmas_samples).shape)
    #print((multinomial_samples*sigmas_samples.unsqueeze(-1)).shape)
    if len(sigmas_samples.shape)==1:
        sigmas_samples=sigmas_samples.unsqueeze(-1)
    out_samples = mus + multinomial_samples*sigmas_samples
    return out_samples
'''
def get_single_gaussian_posterior(mean, sigma, forward_mat, noise_sigma, y):
    ATA = torch.mm(forward_mat.T,forward_mat)/noise_sigma[0,0]
    cov_gauss = torch.linalg.inv(torch.linalg.inv(sigma)+ATA)

    mean_gauss = cov_gauss.mm((forward_mat.T).mm(y.view(DIMENSION,1)/noise_sigma[0,0]))+cov_gauss.mm(torch.linalg.inv(sigma)).mm(mean.view(DIMENSION,1))
    return mean_gauss.view(DIMENSION), cov_gauss



def get_log_single_affiliation_weight(mu, sigma, new_mu, new_sigma, noise_sigma, y):
    log_ck = -0.5 * y.dot(torch.linalg.inv(noise_sigma).mv(y))
    log_ck += -0.5 * mu.dot(torch.linalg.inv(sigma).mv(mu))
    log_ck += 0.5 * new_mu.dot(torch.linalg.inv(new_sigma).mv(new_mu))
    return log_ck.cpu().numpy().astype(np.float64)

def get_mixture_posterior(x_gauss_mixture_params, forward_mat, noise_sigma, y):
    out_mixtures = []
    nenner = 0
    things = []
    for w, mu, sigma in x_gauss_mixture_params:
        mu_new, sigma_new = get_single_gaussian_posterior(mu, sigma, forward_mat, noise_sigma, y)
        things.append((0.5*torch.mm(mu_new.view(DIMENSION,1).T,torch.linalg.inv(sigma_new)).mm(mu_new.view(DIMENSION,1))-0.5*torch.mm(mu.view(DIMENSION,1).T,
                                         torch.linalg.inv(sigma).mm(mu.view(DIMENSION,1))))[0][0].cpu().data.numpy())
    maxi = torch.tensor(max(things))

    for w, mu, sigma in x_gauss_mixture_params:
        mu_new, sigma_new = get_single_gaussian_posterior(mu, sigma, forward_mat, noise_sigma, y)
        #nenner += w*(np.exp(0.5*torch.mm(mu_new.view(DIMENSION,1).T,torch.linalg.inv(sigma_new)).mm(mu_new.view(DIMENSION,1))[0][0].cpu().data.numpy()))
        nenner += w*(torch.exp(0.5*torch.mm(mu_new.view(DIMENSION,1).T,
                                         torch.linalg.inv(sigma_new).mm(mu_new.view(DIMENSION,1)))-0.5*torch.mm(mu.view(DIMENSION,1).T,
                                         torch.linalg.inv(sigma).mm(mu.view(DIMENSION,1))))[0,0].cpu().data.numpy())


    for w, mu, sigma in x_gauss_mixture_params:
        mu_new, sigma_new = get_single_gaussian_posterior(mu, sigma, forward_mat, noise_sigma, y)
        out_mixtures.append((w*(torch.exp(0.5*torch.mm(mu_new.view(DIMENSION,1).T,
                                         torch.linalg.inv(sigma_new).mm(mu_new.view(DIMENSION,1)))-0.5*torch.mm(mu.view(DIMENSION,1).T,
                                         torch.linalg.inv(sigma).mm(mu.view(DIMENSION,1))))[0,0].cpu().data.numpy())/nenner,
                            mu_new, sigma_new))

    return out_mixtures

def get_mix_density(mixture_params, samples):
    density = 0
    for i in range(len(samples)):
        for (w,mu, sigma) in mixture_params:
            mu = mu.cpu().data.numpy()
            sigma = sigma.cpu().data.numpy()
            density += np.exp(-0.5*np.matmul(np.matmul((samples[i]-mu).T,np.linalg.inv(sigma)),(samples[i]-mu)))/(np.sqrt(np.linalg.det(sigma)*(2*np.pi)**DIMENSION))
    return density/len(samples)

def cross_entropy(samples,mix_params_post):
    return -get_mix_density(mix_params_post, samples)



def create_forward_model(scale):
    s = scale*torch.abs(torch.ones(DIMENSION, device = device))

    return torch.diag(s)

def forward_pass(x, forward_map):
    return torch.matmul(forward_map, x.T).T



def get_prior_log_likelihood(samples, mixture_params, eps=1e-4):
    exponent_difference = -np.log(eps)
    exponents = torch.zeros((samples.shape[0], len(mixture_params)), device=device)
    for k, (w, mu, sigma) in enumerate(mixture_params):
        log_gauss_prefactor = np.log(2 * np.pi) * (-DIMENSION / 2) + DIMENSION*(torch.log(torch.tensor(0.5)))*(-0.5)
        tmp = -0.5 * torch.sum((samples - mu[None, :]) * (20*((samples - mu[None, :]).T)).T, dim=1)
        exponents[:, k] = tmp + np.log(w) + log_gauss_prefactor

    max_exponent = torch.max(exponents, dim=1)[0].detach()
    exponent_mask = exponents.clone().detach() >= (max_exponent - exponent_difference)[:, None]
    summed_exponents = torch.sum(torch.where(exponent_mask, torch.exp(exponents - max_exponent[:, None]), torch.zeros_like(exponents)), dim=1)
    return torch.log(summed_exponents) + max_exponent


def get_prior_log_likelihood_general(samples, mixture_params, eps=1e-4):
    exponent_difference = -np.log(eps)
    exponents = torch.zeros((samples.shape[0], len(mixture_params)), device=device)
    for k, (w, mu, sigma) in enumerate(mixture_params):
        sigma_inv = torch.linalg.inv(sigma)
        log_gauss_prefactor = np.log(2 * np.pi) * (-DIMENSION / 2) + DIMENSION*(torch.log(torch.tensor(0.5)))*(-0.5)
        tmp = -0.5 * torch.sum((samples - mu[None, :]) * (sigma_inv.mm((samples - mu[None, :]).T)).T, dim=1)
        exponents[:, k] = tmp + np.log(w) + log_gauss_prefactor

    max_exponent = torch.max(exponents, dim=1)[0].detach()
    exponent_mask = exponents.clone().detach() >= (max_exponent - exponent_difference)[:, None]
    summed_exponents = torch.sum(torch.where(exponent_mask, torch.exp(exponents - max_exponent[:, None]), torch.zeros_like(exponents)), dim=1)
    return torch.log(summed_exponents) + max_exponent





def get_prior_log_likelihood_naive(samples, mixture_params):
        densities = torch.zeros(samples.shape[0], device=device)

        for w, mu, sigma in mixture_params:
            sigma_inv = torch.linalg.inv(sigma)
            gauss_prefactor = (2 * np.pi) ** (-DIMENSION / 2) * torch.linalg.det(sigma) ** (-0.5)
            tmp = -0.5 * torch.sum((samples - mu[None, :]) * (sigma_inv.mm((samples - mu[None, :]).T)).T, dim=1)
            densities += w * torch.exp(tmp) * gauss_prefactor

        return torch.log(densities)

def get_log_posterior(samples, forward_map, mixture_params, b, y):
    p = -get_prior_log_likelihood(samples, mixture_params)
    p2 = 0.5 * torch.sum((y-forward_pass(samples, forward_map))**2 * (1/b**2), dim=1)
    return (p+p2).view(len(samples))



def get_epoch_data_loader(mixture_params, num_samples_per_epoch, batch_size, forward_map, b):
    x = draw_mixture_dist(mixture_params, num_samples_per_epoch)
    y = forward_pass(x, forward_map)
    y += torch.randn_like(y) * b
    def epoch_data_loader():
        for i in range(0, num_samples_per_epoch, batch_size):
            yield x[i:i+batch_size].clone(), y[i:i+batch_size].clone()

    return epoch_data_loader



def make_image(true_samples, pred_samples, img_name):
    cmap = plt.cm.tab20
    range_param = 1
    no_params = min(5, DIMENSION)
    fig, axes = plt.subplots(figsize=[12,12], nrows=no_params, ncols=no_params, gridspec_kw={'wspace':0., 'hspace':0.});

    for j in range(no_params):
        for k in range(no_params):
            axes[j,k].get_xaxis().set_ticks([])
            axes[j,k].get_yaxis().set_ticks([])
            # if k == 0: axes[j,k].set_ylabel(j)
            # if j == len(params)-1: axes[j,k].set_xlabel(k);
            if j == k:
                axes[j,k].hist(pred_samples[:,j], bins=50, color=cmap(0), alpha=0.3, range=(-range_param,range_param))
                axes[j,k].hist(pred_samples[:,j], bins=50, color=cmap(0), histtype="step", range=(-range_param,range_param))

                axes[j,k].hist(true_samples[:,j], bins=50, color=cmap(2), alpha=0.3, range=(-range_param,range_param))
                axes[j,k].hist(true_samples[:,j], bins=50, color=cmap(2), histtype="step", range=(-range_param,range_param))
            else:
                val, x, y = np.histogram2d(pred_samples[:,j], pred_samples[:,k], bins=25, range = [[-range_param, range_param], [-range_param, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(0)])

                val, x, y = np.histogram2d(true_samples[:,j], true_samples[:,k], bins=25, range = [[-range_param, range_param], [-range_param, range_param]])
                axes[j,k].contour(val, 8, extent=[x[0], x[-1], y[0], y[-1]], alpha=0.5, colors=[cmap(2)])

    plt.savefig('./Images/'+img_name)
    plt.close()

##################################################################################################
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

def train_inn_epoch(optimizer, model, epoch_data_loader):
    mean_loss = 0
    for k, (x, y) in enumerate(epoch_data_loader()):
        cur_batch_size = len(x)

        loss = 0
        invs, jac_inv = model(x, c = y, rev = True)

        l5 = 0.5 * torch.sum(invs**2, dim=1) - jac_inv
        loss += (torch.sum(l5) / cur_batch_size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        mean_loss = mean_loss * k / (k + 1) + loss.data.item() / (k + 1)
    return mean_loss
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



