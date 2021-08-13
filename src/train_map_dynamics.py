import numpy as np
import argparse
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from Experiments.GetTestParameters import GetTest1Parameters


def action_obs_1d_ind(nu, no, actions, obs):
    K = np.arange(nu*no).reshape((nu, no))
    ind = []
    for i in range(len(actions)):
        ind.append(K[actions[i], obs[i]])
    return ind


def cal_loss(B_model, r_model, D_pre_model, loss_fn_z, loss_fn_r, nu, no, bt, bp, b_next, actions, action_obs_ind, r,
             l=1, tau=1, B_det_model=None, P_o_za_model=None, P_o_ba=None):
    pred_loss = 0
    r_loss = 0
    obs_loss = 0
    for i in range(nu):
        # Calculate loss for each (discrete) action
        ind = (actions == i)
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


def minimize_AIS(D_pre_model, nu, nz, bt, bp, actions, tau=1):
    z_list = np.zeros(nz)
    for i in range(nu):
        # Calculate loss for each (discrete) action
        ind = (actions == i)
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
    # State & belief collection is shifted by one time index from a, r, o, P_o_ba
    a.pop(0)
    r.pop(0)
    o.pop(0)
    P_o_ba.pop(0)
    g_dim = BO[0].g[0].dim
    input_dim = ncBelief * (1 + g_dim + g_dim ** 2)  # w, mean, flatten(Sigma)
    bt = []
    b_next = []
    b_next_p = []
    st = []
    s_next = []
    action_indices = []
    observation_indices = []
    reward = []
    P_o_ba_t = []
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
                if i < num_samples - 1:
                    b_next_p.append(bp)
                    s_next.append(s[i+1])
                    action_indices.append(int(a[i] - 1))
                    observation_indices.append(int(o[i]))
                    reward.append(r[i])
                    P_o_ba_t.append(P_o_ba[i])
                if i > 0 and i not in step_ind_copy:
                    b_next.append(bt[-1])
    else:
        pass

    return np.array(bt[:-1]), np.array(b_next), np.array(b_next_p), np.array(st[:-1]), np.array(s_next), input_dim, \
           g_dim, action_indices, observation_indices, np.array(reward), np.array(P_o_ba_t)


def save_data(input_dim, st_, s_next_, bt_, bp_, b_next_, action_indices, action_obs_ind, r_, P_o_ba_t_):
    folder_name = "data/"
    save_dict = {"input_dim": input_dim, "bt": bt_, "bp": bp_, "b_next": b_next_, "action_indices": action_indices,
                 "action_obs_ind": action_obs_ind, "r": r_, "st": st_, "s_next": s_next_, "P_o_ba": P_o_ba_t_}
    torch.save(save_dict, folder_name + "data_tensor_w_state.pth")


def load_data(file_name="data/data_tensor_w_state.pth"):
    load_dict = torch.load(file_name)
    return load_dict["input_dim"], load_dict["st"], load_dict["s_next"],load_dict["bt"], load_dict["bp"], \
           load_dict["b_next"], load_dict["action_indices"], load_dict["action_obs_ind"], load_dict["r"], \
           load_dict["P_o_ba"]


def save_model(B_model, r_model, D_pre_model, z_list, nz, nf, tau, B_det_model=None, P_o_za_model=None):
    folder_name = "model/" + "s_fit/"

    r_dict = {}
    if B_det_model is not None:
        folder_name += "AP2ab/" + "obs_l_weight_2/"
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
    folder_name = "model/" + "s_fit/"
    if AP2ab:
        B = np.load(folder_name + "B_det_{}_{}_{}.npy".format(nz, nf, tau))
    else:
        B = np.load(folder_name + "B_{}_{}_{}.npy".format(nz, nf, tau))
    z_list = np.load(folder_name + "zList_{}_{}_{}.npy".format(nz, nf, tau))
    # z_list = np.arange(nz)
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
    parser.add_argument("--tau", help="Temperature for Gumbel Softmax", type=float, default=1)
    parser.add_argument("--nz", help="Number of Discrete AIS", type=int, default=100)
    parser.add_argument("--num_epoch", help="Number of Samples for Belief", type=int, default=10000)
    parser.add_argument("--num_samples", help="Number of Training Samples", type=int, default=10000)
    parser.add_argument("--resume_training", help="Resume training for the model", action="store_true")
    parser.add_argument("--generate_data", help="Generate belief samples", action="store_true")
    parser.add_argument("--scheduler", help="Set StepLR scheduler", action="store_true")
    args = parser.parse_args()

    # Sample belief states data
    ncBelief = 10
    POMDP, P = GetTest1Parameters(ncBelief=ncBelief)
    num_samples = args.num_samples

    nz = args.nz
    nu = 3
    no = 4
    tau = args.tau
    AP2ab = False

    writer = SummaryWriter()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.generate_data:
        BO, BS, s, a, o, r, P_o_ba, step_ind = POMDP.SampleBeliefs(P["start"], num_samples, P["dBelief"],
                                                                   P["stepsXtrial"], P["rMin"], P["rMax"],
                                                                   obs_prob=args.pred_obs)
        bt, b_next, bp, st, s_next, input_dim, g_dim, action_indices, observation_indices, reward, P_o_ba_t = \
            process_belief(BO, BS, num_samples, step_ind, ncBelief, s, a, o, r, P_o_ba)
        action_obs_ind = action_obs_1d_ind(nu, no, action_indices, observation_indices)
        # Shuffle data
        ind = np.arange(st.shape[0])
        np.random.shuffle(ind)
        st_ = torch.from_numpy(st[ind]).view(st.shape[0], 1).to(torch.float32).to(device)
        s_next_ = torch.from_numpy(s_next[ind]).view(s_next.shape[0], 1).to(torch.float32).to(device)
        bt_ = torch.from_numpy(bt[ind]).to(torch.float32).to(device)
        bp_ = torch.from_numpy(bp[ind]).to(torch.float32).to(device)
        b_next_ = torch.from_numpy(b_next[ind]).to(torch.float32).to(device)
        r_ = torch.from_numpy(reward[ind]).view(st.shape[0], 1).to(torch.float32).to(device)
        P_o_ba_t_ = torch.from_numpy(P_o_ba_t[ind]).to(torch.float32).to(device)
        action_indices = torch.from_numpy(np.array(action_indices)[ind]).to(torch.float32).to(device)
        action_obs_ind = torch.from_numpy(np.array(action_obs_ind)[ind]).to(torch.float32).to(device)
        save_data(input_dim, st_, s_next_, bt_, bp_, b_next_, action_indices, action_obs_ind, r_, P_o_ba_t_)
    else:
        input_dim, st_, s_next_, bt_, bp_, b_next_, action_indices, action_obs_ind, r_, P_o_ba_t_ = load_data()

    nf = 96
    B_model = []
    r_model = []
    for i in range(nu):
        B_model.append(nn.Linear(nz, nz, bias=False).to(device))
        r_model.append(nn.Sequential(
            nn.Linear(nz, nf), nn.LeakyReLU(0.1),
            # nn.Linear(nf, nf), nn.LeakyReLU(0.1),
            nn.Linear(nf, 1)).to(device))
    project_col_sum(B_model)
    D_pre_model = nn.Sequential(
            nn.Linear(1, nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            nn.Linear(nf, 2 * nf), nn.LeakyReLU(0.1),  # nn.ReLU(),
            nn.Linear(2 * nf, 4 * nf), nn.LeakyReLU(0.1),
            nn.Linear(4 * nf, 2 * nf), nn.LeakyReLU(0.1),
            nn.Linear(nf * 2, nf), nn.LeakyReLU(0.1),
            nn.Linear(nf, nz))
    loss_fn_z = nn.L1Loss()  # CrossEntropyLoss()
    loss_fn_r = nn.MSELoss()
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
        B, r_model, D_pre_model, z_list = load_model(nz, nf, nu, tau, AP2ab=AP2ab)
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

    num_epoch = args.num_epoch

    for epoch in range(num_epoch):
        optimizer.zero_grad()
        pred_loss, r_loss, obs_loss = cal_loss(B_model, r_model, D_pre_model, loss_fn_z, loss_fn_r, nu, no, st_, s_next_,
                                               b_next_, action_indices, action_obs_ind, r_, l=10, tau=tau,
                                               B_det_model=B_det_model, P_o_za_model=P_o_za_model, P_o_ba=P_o_ba_t_)
        loss = pred_loss + r_loss + obs_loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Projected Gradient Descent to ensure column sum of B = 1
        project_col_sum(B_model)
        if B_det_model is not None:
            project_col_sum(B_det_model)
        if P_o_za_model is not None:
            project_col_sum(P_o_za_model)
        if epoch % 100 == 0:
            print(epoch)
            print("Prediction loss: {}, reward loss: {}, observation loss: {}".format(pred_loss, r_loss, obs_loss))
        writer.add_scalar("Loss/{}_sample_{}_nz".format(num_samples, nz), loss, epoch)
    writer.flush()

    z_list = minimize_AIS(D_pre_model, nu, nz, st_, s_next_, action_indices, tau=tau)
    D_pre_model.cpu()
    save_model(B_model, r_model, D_pre_model, z_list, nz, nf, tau, B_det_model, P_o_za_model)


