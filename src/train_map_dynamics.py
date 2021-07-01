import math
import numpy as np
import argparse
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from Experiments.GetTestParameters import GetTest1Parameters


def action_obs_1d_ind(nu, no, actions, obs):
    K = np.arange(nu*no).reshape((nu, no))
    ind = []
    for i in range(len(actions)):
        ind.append(K[actions[i], obs[i]])
    return ind


def cal_loss(B_model, r_model, D_pre_model, loss_fn_z, loss_fn_r, nu, no, data,
             l=1, tau=1, B_det_model=None, P_o_za_model=None):
    bt, bp, b_next, r, P_o_ba, action_ind, action_obs_ind = data
    pred_loss = 0
    r_loss = 0
    obs_loss = 0
    for i in range(nu):
        # Calculate loss for each (discrete) action
        ind = (action_ind == i)
        if len(bt[ind]) > 0:
            Db = F.gumbel_softmax(D_pre_model(bt[ind]), tau=tau, hard=True)
            if B_det_model is None:
                z = B_model[i](Db)
                z_next = F.gumbel_softmax(D_pre_model(bp[ind]), tau=tau, hard=True)
                # Obtain class for cross entropy loss
                # _, z_next_class = z_next.max(dim=1)
                pred_loss += loss_fn_z(z, z_next)
            r_pred = r_model[i](Db)
            r_loss += l*loss_fn_r(r_pred, r[ind])
            if P_o_za_model is not None:
                obs_pred = P_o_za_model[i](Db)
                obs_loss += l * loss_fn_r(obs_pred, P_o_ba[ind])
    if B_det_model is not None:
        for i in range(nu * no):
            ind = (action_obs_ind == i)
            Db = F.gumbel_softmax(D_pre_model(bt[ind]), tau=tau, hard=True)
            z_det = B_det_model[i](Db)
            z_next_o = F.gumbel_softmax(D_pre_model(b_next), tau=tau, hard=True)
            pred_loss += loss_fn_z(z_det, z_next_o[ind])

    return pred_loss, r_loss, obs_loss


def hard_negative_sample(B_model, r_model, D_pre_model, z_dist, r_dist, nu, no, mining_loader, mining_ratio, batch_size,
                         l=10, tau=1, B_det_model=None, P_o_za_model=None):
    for i, data in enumerate(mining_loader, 0):
        bt, bp, b_next, r, P_o_ba, action_ind, action_obs_ind = data
    index = []
    loss = torch.tensor([]).to(device)
    for i in range(nu):
        # Calculate loss for each (discrete) action
        ind = np.where(action_ind.cpu() == i)[0]
        index += list(ind)
        Db = F.gumbel_softmax(D_pre_model(bt[ind]), tau=tau, hard=True)
        if B_det_model is None:
            z = B_model[i](Db)
            z_next = F.gumbel_softmax(D_pre_model(bp[ind]), tau=tau, hard=True)
            pred_loss = z_dist(z, z_next)
        r_pred = r_model[i](Db)
        r_loss = r_dist(r_pred, r[ind])
        loss = torch.cat((loss, pred_loss + l*r_loss))
    index = torch.tensor(index)
    sort_ind = torch.argsort(loss, descending=True)
    pick_ind = index[sort_ind][:int(np.ceil(mining_ratio*len(loss)))]
    data_set = TensorDataset(bt[pick_ind], bp[pick_ind], b_next[pick_ind], r[pick_ind], P_o_ba[pick_ind],
                             action_ind[pick_ind], action_obs_ind[pick_ind])
    return DataLoader(data_set, batch_size=batch_size)


def minimize_AIS(D_pre_model, nu, nz, bt, bp, action_ind, tau=1):
    z_list = np.zeros(nz)
    for i in range(nu):
        # Calculate loss for each (discrete) action
        ind = (action_ind == i)
        Db = F.gumbel_softmax(D_pre_model(bt[ind]), tau=tau, hard=True).cpu().detach().numpy()
        z_next = F.gumbel_softmax(D_pre_model(bp[ind]), tau=tau, hard=True).cpu().detach().numpy()
        z_list += np.sum(Db+z_next, axis=0)
    z_ind = np.arange(nz)
    return z_ind[z_list > 0]


def fit_state_loss(r_model_u, loss_fn, s, r, actions, nu):
    loss = 0
    for i in range(nu):
        ind = (actions == i)
        r_pred = r_model_u[i](s[ind])
        loss += loss_fn(r_pred, r[ind])
    return loss


def project_col_sum(model):
    # Shift weights to nonnegative and scale for col sum to be 1
    for i in range(len(model)):
        m = model[i]
        d = m.weight.data
        n = m.weight.shape[1]
        m, _ = torch.min(d, axis=0)
        d = d - m.view([1, n])
        s = torch.sum(d, 0).view([1, n])
        model[i].weight.data = d/s


def process_belief(BO, B, num_samples, step_ind, ncBelief, s, a, o, r, P_o_ba):
    step_ind_copy = np.array(step_ind)
    # Remove the first belief before observation (useless)
    B.pop(0)
    g_dim = BO[0].g[0].dim
    input_dim = ncBelief * (1 + g_dim + g_dim ** 2)  # w, mean, flatten(Sigma)
    bt = []
    b_next = []
    b_next_p = []
    st = []
    s_next = []
    reward = []
    P_o_ba_t = []
    action_indices = []
    observation_indices = []
    if g_dim == 1:
        for i in range(num_samples):
            b = BO[i].to_array()
            # Belief before observation
            if i < num_samples-1:
                bp = B[i].to_array()
            if i in step_ind_copy-1:
                if i not in step_ind_copy:
                    b_next.append(b)
            else:
                bt.append(b)
                st.append(s[i])
                action_indices.append(int(a[i] - 1))
                observation_indices.append(int(o[i]))
                reward.append(r[i])
                P_o_ba_t.append(P_o_ba[i])
                if i < num_samples - 1:
                    b_next_p.append(bp)
                    s_next.append(s[i+1])
                if i > 0 and i not in step_ind_copy:
                    b_next.append(bt[-1])
    else:
        pass

    return np.array(bt[:-1]), np.array(b_next), np.array(b_next_p), np.array(st[:-1]), np.array(s_next), \
           np.array(reward[:-1]), np.array(P_o_ba_t[:-1]), action_indices[:-1], observation_indices[:-1],\
           input_dim, g_dim,


def save_data(input_dim, mining_loader):
    folder_name = "data/"
    save_dict = {"input_dim": input_dim, "mining_loader": mining_loader}
    torch.save(save_dict, folder_name + "mining_loader_data.pth")


def load_data(file_name="data/mining_loader_data.pth"):
    load_dict = torch.load(file_name)
    return load_dict["input_dim"], load_dict["mining_loader"]


def save_model(B_model, r_model, D_pre_model, z_list, nz, nf, tau, B_det_model=None, P_o_za_model=None):
    folder_name = "model/" + "100k/" + "7_layer/"
    r_dict = {}
    if B_det_model is not None:
        folder_name += "AP2ab/" + "obs_l_weight/"
        B_det = []
        for i in range(len(B_det_model)):
            B_det_model[i].cpu()
            B_det.append(B_det_model[i].weight.data.numpy())
        np.save(folder_name + "B_det_{}_{}_{}".format(nz, nf, tau), B_det)
        for j in range(len(r_model)):
            r_model[j].cpu()
            r_dict[str(j)] = r_model[j].state_dict()
            r_dict["model_" + str(j)] = r_model[j]
    else:
        B = []
        for i in range(len(B_model)):
            B_model[i].cpu()
            B.append(B_model[i].weight.data.numpy())
            r_model[i].cpu()
            r_dict[str(i)] = r_model[i].state_dict()
            r_dict["model_" + str(i)] = r_model[i]
        np.save(folder_name + "B_{}_{}_{}".format(nz, nf, tau), B)
    if P_o_za_model is not None:
        P_o_za = []
        for i in range(len(P_o_za_model)):
            P_o_za_model[i].cpu()
            P_o_za.append(P_o_za_model[i].weight.data.numpy())
        np.save(folder_name + "P_o_za_{}_{}_{}".format(nz, nf, tau), P_o_za)

    np.save(folder_name + "zList_{}_{}_{}".format(nz, nf, tau), z_list)
    torch.save(r_dict, folder_name + "r_{}_{}_{}.pth".format(nz, nf, tau))
    torch.save(D_pre_model.state_dict(), folder_name + "D_pre_{}_{}_{}.pth".format(nz, nf, tau))
    torch.save(D_pre_model, folder_name + "D_pre_{}_{}_{}_model.pth".format(nz, nf, tau))


def load_model(nz, nf, nu, tau, AP2ab=False):
    folder_name = "model/" + "100k/" + "6_layer/" #+ "scheduler/"
    if AP2ab:
        B = np.load(folder_name + "B_det_{}_{}_{}.npy".format(nz, nf, tau))
    else:
        B = np.load(folder_name + "B_{}_{}_{}.npy".format(nz, nf, tau))
    z_list = np.load(folder_name + "zList_{}_{}_{}.npy".format(nz, nf, tau))
    r_dict = torch.load(folder_name + "r_{}_{}_{}.pth".format(nz, nf, tau))
    r = []
    for i in range(nu):
        r.append(r_dict["model_" + str(i)])
        r[i].load_state_dict(r_dict[str(i)])
        r[i].eval()
    D = torch.load(folder_name + "D_pre_{}_{}_{}_model.pth".format(nz, nf, tau))
    D.load_state_dict(torch.load(folder_name + "D_pre_{}_{}_{}.pth".format(nz, nf, tau)))
    D.eval()
    return B, r, D, z_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_trans", help="Fit the deterministic transition of AIS (AP2a)", action="store_true")
    parser.add_argument("--pred_obs", help="Predict the observation (AP2b)", action="store_true")
    parser.add_argument("--generate_data", help="Generate belief samples", action="store_true")
    parser.add_argument("--tau", help="Temperature for Gumbel Softmax", type=float, default=1)
    parser.add_argument("--mining_step", help="Steps to remove easy samples and add hard negatives", type=int,
                        default=500)
    parser.add_argument("--mining_ratio", help="Number of hard negative samples for training", type=float, default=0.5)
    parser.add_argument("--batch_size", help="Training batch size", type=int, default=1000)
    parser.add_argument("--resume_training", help="Resume training for the model", action="store_true")
    parser.add_argument("--scheduler", help="Set StepLR scheduler", action="store_true")
    args = parser.parse_args()

    # Sample belief states data
    ncBelief = 10  #5
    POMDP, P = GetTest1Parameters(ncBelief=ncBelief)
    num_samples = 100000
    nz = 1000
    nu = 3
    no = 4
    tau = args.tau

    # "End-to-end" training to minimize AP12
    writer = SummaryWriter()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.generate_data:
        BO, BS, s, a, o, r, P_o_ba, step_ind = POMDP.SampleBeliefs(P["start"], num_samples, P["dBelief"],
                                                                   P["stepsXtrial"], P["rMin"], P["rMax"],
                                                                   obs_prob=args.pred_obs)
        bt, b_next, bp, st, s_next, reward, P_o_ba_t, action_ind, observation_ind, input_dim, g_dim = \
            process_belief(BO, BS, num_samples, step_ind, ncBelief, s, a, o, r, P_o_ba)
        action_obs_ind = action_obs_1d_ind(nu, no, action_ind, observation_ind)

        st_ = torch.from_numpy(st).view(st.shape[0], 1).to(torch.float32).to(device)
        s_next_ = torch.from_numpy(s_next).view(s_next.shape[0], 1).to(torch.float32).to(device)
        bt_ = torch.from_numpy(bt).to(torch.float32).to(device)
        bp_ = torch.from_numpy(bp).to(torch.float32).to(device)
        b_next_ = torch.from_numpy(b_next).to(torch.float32).to(device)
        r_ = torch.from_numpy(reward).view(reward.shape[0], 1).to(torch.float32).to(device)
        P_o_ba_t_ = torch.from_numpy(P_o_ba_t).to(torch.float32).to(device)
        action_ind = torch.tensor(action_ind).to(device)
        action_obs_ind = torch.tensor(action_obs_ind).to(device)

        train_set = TensorDataset(bt_, bp_, b_next_, r_, P_o_ba_t_, action_ind, action_obs_ind)
        mining_loader = DataLoader(train_set, batch_size=num_samples, shuffle=True)
        save_data(input_dim, mining_loader)
    else:
        input_dim, mining_loader = load_data()
        for i, mining_data in enumerate(mining_loader, 0):
            bt_, bp_, _, r_, _, action_ind, _ = mining_data

    nf = 96
    B_model = []
    r_model = []
    for i in range(nu):
        B_model.append(nn.Linear(nz, nz, bias=False).to(device))
        r_model.append(nn.Sequential(
            nn.Linear(nz, nf), nn.LeakyReLU(0.1),
            nn.Linear(nf, 1)).to(device))
    project_col_sum(B_model)
    D_pre_model = nn.Sequential(
            nn.Linear(input_dim, nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            nn.Linear(nf, 2 * nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            nn.Linear(2 * nf, 4 * nf), nn.LeakyReLU(0.1),
            nn.Linear(4 * nf, 4 * nf), nn.LeakyReLU(0.1),
            nn.Linear(4 * nf, 2 * nf), nn.LeakyReLU(0.1),
            nn.Linear(nf * 2, nf), nn.LeakyReLU(0.1),
            nn.Linear(nf, nz))
    loss_fn_z = nn.L1Loss()  # CrossEntropyLoss()
    loss_fn_r = nn.MSELoss()
    z_dist = nn.PairwiseDistance(p=1)
    r_dist = nn.PairwiseDistance(p=2)
    D_pre_model.to(device)
    B_det_model = None
    if args.det_trans:
        AP2ab = True
        B_det_model = []
        for i in range(nu * no):
            B_det_model.append(nn.Linear(nz, nz, bias=False).to(device))
        project_col_sum(B_det_model)
    P_o_za_model = None
    if args.pred_obs:
        P_o_za_model = []
        for i in range(nu):
            P_o_za_model.append(nn.Linear(nz, no, bias=False).to(device))
        project_col_sum(P_o_za_model)

    if args.resume_training:
        B, r_model, D_pre_model, z_list = load_model(nz, nf, nu, tau, AP2ab=args.det_trans)
        D_pre_model.to(device)
        for i in range(len(r_model)):
            r_model[i].to(device)

    params = list(D_pre_model.parameters())
    if B_det_model is None:
        for x in B_model:
            params += x.parameters()
    else:
        for x in B_det_model:
            params += x.parameters()
    if P_o_za_model is not None:
        for x in P_o_za_model:
            params += x.parameters()
    for x in r_model:
        params += x.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3)

    scheduler = None
    if args.scheduler:
        scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)

    num_epoch = 20000

    data_loader = hard_negative_sample(B_model, r_model, D_pre_model, z_dist, r_dist, nu, no, mining_loader,
                                       args.mining_ratio, args.batch_size, l=10, tau=tau, B_det_model=B_det_model,
                                       P_o_za_model=P_o_za_model)

    for epoch in range(num_epoch):
        for i, data in enumerate(data_loader, 0):
            optimizer.zero_grad()
            pred_loss, r_loss, obs_loss = cal_loss(B_model, r_model, D_pre_model, loss_fn_z, loss_fn_r, nu, no, data,
                                                   l=10, tau=tau, B_det_model=B_det_model, P_o_za_model=P_o_za_model)
            loss = pred_loss + r_loss + obs_loss
            loss.backward()
            optimizer.step()
            # Projected Gradient Descent to ensure column sum of B = 1
            project_col_sum(B_model)
            if B_det_model is not None:
                project_col_sum(B_det_model)
            if P_o_za_model is not None:
                project_col_sum(P_o_za_model)

        if scheduler is not None:
            scheduler.step()
        if epoch % args.mining_step == 0:
            data_loader = hard_negative_sample(B_model, r_model, D_pre_model, z_dist, r_dist, nu, no, mining_loader,
                                               args.mining_ratio, args.batch_size, l=10, tau=tau, B_det_model=B_det_model,
                                               P_o_za_model=P_o_za_model)
        if epoch % 100 == 0:
            print(epoch)
            print("Prediction loss: {}, reward loss: {}, observation loss: {}".format(pred_loss, r_loss, obs_loss))
        writer.add_scalar("Loss/{}_sample_{}_nz".format(num_samples, nz), loss, epoch)
    writer.flush()

    z_list = minimize_AIS(D_pre_model, nu, nz, bt_, bp_, action_ind, tau=tau)
    D_pre_model.cpu()
    save_model(B_model, r_model, D_pre_model, z_list, nz, nf, tau, B_det_model, P_o_za_model)


