import math
import numpy as np
import argparse
import torch
from torch import Tensor
from torch.nn.parameter import Parameter #, UninitializedParameter
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules import Module
from Experiments.GetTestParameters import GetTest1Parameters

class nD_Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
        This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
        Args:
            tensor_features: size of first dimension of tensor
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``
        Shape:
            - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
              additional dimensions and :math:`H_{in} = \text{in\_features}`
            - Output: :math:`(N, tensor_feature, *, H_{out})` where all but the last dimension
              are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
        Attributes:
            weight: the learnable weights of the module of shape
                :math:`(\text{out\_features}, \text{in\_features})`. The values are
                initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in\_features}}`
            bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                    If :attr:`bias` is ``True``, the values are initialized from
                    :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                    :math:`k = \frac{1}{\text{in\_features}}`
    """
    __constants__ = ['in_features', 'out_features']
    tensor_features: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, tensor_features: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(nD_Linear, self).__init__()
        self.tensor_features = tensor_features
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(tensor_features, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(tensor_features, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            return (self.weight@input.T).permute(2,0,1)+ self.bias
        else:
            return (self.weight@input.T).permute(2,0,1)

    def extra_repr(self) -> str:
        return 'tensor_features = {}, in_features={}, out_features={}, bias={}'.format(
            self.tensor_features, self.in_features, self.out_features, self.bias is not None
        )


def action_obs_1d_ind(nu, no, actions, obs):
    K = np.arange(nu*no).reshape((nu, no))
    ind = []
    for i in range(len(actions)):
        ind.append(K[actions[i], obs[i]])
    return ind


def cal_loss(B_model, r_model, D_pre_model, loss_fn, bt, bp, b_next, actions, action_obs_ind, r, l=1, B_det_model=None):
    Db = F.gumbel_softmax(D_pre_model(bt), hard=True)
    z = B_model(Db)
    z_next = F.gumbel_softmax(D_pre_model(bp), hard=True)
    r_pred = r_model(Db)
    pred_loss = loss_fn(z[torch.arange(len(actions)), actions, :], z_next)
    r_loss = l*loss_fn(r_pred[torch.arange(len(actions)), actions], r)
    if B_det_model is not None:
        z_det = B_det_model(Db)
        z_next_o = F.gumbel_softmax(D_pre_model(b_next), hard=True)
        pred_loss += loss_fn(z_det[torch.arange(len(action_obs_ind)), action_obs_ind, :], z_next_o)
    return pred_loss, r_loss


def project_col_sum(model):
    model.weight.data.clamp_(min=0, max=1)
    s = torch.sum(model.weight, 1).view([model.weight.shape[0], 1, model.weight.shape[2]])
    model.weight.data = model.weight.data/s


def process_belief(BO, B, num_samples, step_ind, ncBelief, a, o, r):
    step_ind_copy = np.array(step_ind)
    # Remove the first belief before observation (useless)
    B.pop(0)
    g_dim = BO[0].g[0].dim
    input_dim = ncBelief * (1 + g_dim + g_dim ** 2)  # w, mean, flatten(Sigma)
    bt = []
    b_next = []
    b_next_p = []
    action_indices = []
    observation_indices = []
    reward = []
    if g_dim == 1:
        for i in range(num_samples):
            b_object = BO[i]
            # If the belief object has mixtures fewer than ncBelief, fill with zeros
            b = np.zeros(input_dim)
            nBelief = len(b_object.w)
            if nBelief == 0:
                print("Empty belief, index: ", i)
            b[:nBelief] = b_object.w
            b[ncBelief:ncBelief + nBelief] = [g.m for g in b_object.g]
            b[ncBelief * (g_dim + 1):ncBelief * (g_dim + 1) + nBelief] = [g.S for g in b_object.g]
            # Belief before observation
            if i < num_samples-1:
                bp_object = B[i]
                bp = np.zeros(input_dim)
                npBelief = len(bp_object.w)
                bp[:npBelief] = bp_object.w
                bp[ncBelief:ncBelief + npBelief] = [g.m for g in bp_object.g]
                bp[ncBelief * (g_dim + 1):ncBelief * (g_dim + 1) + npBelief] = [g.S for g in bp_object.g]
            if i in step_ind_copy-1:
                if i not in step_ind_copy:
                    b_next.append(b)
            else:
                bt.append(b)
                if i < num_samples - 1:
                    b_next_p.append(bp)
                action_indices.append(int(a[i]-1))
                observation_indices.append(int(o[i]))
                reward.append(r[i])
                if i > 0 and i not in step_ind_copy:
                    b_next.append(bt[-1])
    else:
        pass

    return np.array(bt[:-1]), np.array(b_next), np.array(b_next_p), input_dim, g_dim, action_indices[:-1], \
           observation_indices[:-1], reward[:-1]


def save_model(B_model, r_model, D_pre_model, nz, nf, B_det_model=None):
    folder_name = "model/"
    if B_det_model is not None:
        folder_name += "det/"
        B_det_model.cpu()
        np.save(folder_name + "B_det_{}_{}".format(nz, nf), B_det_model.weight.data.numpy())
    np.save(folder_name + "B_{}_{}".format(nz, nf), B_model.weight.data.numpy())
    np.save(folder_name + "r_{}_{}".format(nz, nf), r_model.weight.data.numpy())
    torch.save(D_pre_model.state_dict(), folder_name + "D_pre_{}_{}.pth".format(nz, nf))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_trans", help="Fit the deterministic transition of AIS (AP2a)", action="store_true")
    parser.add_argument("--pred_obs", help="Predict the observation (AP2b)", action="store_true")
    args = parser.parse_args()

    # Sample belief states data
    POMDP, P = GetTest1Parameters()
    num_samples = 10000
    ncBelief = 4
    BO, B, s, a, o, r, step_ind = POMDP.SampleBeliefs(P["start"], num_samples, P["dBelief"],
                                                      P["stepsXtrial"], P["rMin"], P["rMax"])
    nz = 30
    nu = 3
    no = 4

    bt, b_next, bp, input_dim, g_dim, action_indices, observation_indices, reward = process_belief(BO, B, num_samples,
                                                                                                   step_ind, ncBelief, a, o, r)
    action_obs_ind = action_obs_1d_ind(nu, no, action_indices, observation_indices)

    # "End-to-end" training to minimize AP12
    writer = SummaryWriter()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    nf = 96
    B_model = nD_Linear(nu, nz, nz, bias=False)
    project_col_sum(B_model)
    r_model = nn.Linear(nz, nu)
    D_pre_model = nn.Sequential(
            nn.Linear(input_dim, nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            nn.Linear(nf, 2 * nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            nn.Linear(2 * nf, nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            # nn.Linear(nf * 2, nf), nn.ReLU(),
            nn.Linear(nf, nz))
    loss_fn = nn.MSELoss()
    B_model.to(device)
    r_model.to(device)
    D_pre_model.to(device)
    B_det_model = None
    if args.det_trans:
        B_det_model = nD_Linear(nu*no, nz, nz, bias=False)
        project_col_sum(B_det_model)
        B_det_model.to(device)

    params = list(B_model.parameters()) + list(r_model.parameters()) + list(D_pre_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    num_epoch = 50000

    bt_ = torch.from_numpy(bt).to(torch.float32).to(device)
    bp_ = torch.from_numpy(bp).to(torch.float32).to(device)
    b_next_ = torch.from_numpy(b_next).to(torch.float32).to(device)
    r_ = torch.from_numpy(np.array(reward)).to(torch.float32).to(device)

    for epoch in range(num_epoch):
        pred_loss, r_loss = cal_loss(B_model, r_model, D_pre_model, loss_fn, bt_, bp_, b_next_, action_indices,
                                     action_obs_ind, r_, B_det_model=B_det_model)
        loss = pred_loss + r_loss
        loss.backward()
        optimizer.step()
        # Projected Gradient Descent to ensure column sum of B = 1
        project_col_sum(B_model)
        if B_det_model is not None:
            project_col_sum(B_det_model)
        if epoch % 100 == 0:
            print(epoch)
            print("Prediction loss: {}, reward loss: {}".format(pred_loss, r_loss))
        writer.add_scalar("Loss/{}_sample_{}_nz".format(num_samples, nz), loss, epoch)
    writer.flush()

    B_model.cpu()
    r_model.cpu()
    D_pre_model.cpu()
    save_model(B_model, r_model, D_pre_model, nz, nf, B_det_model)


