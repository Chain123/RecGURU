import _pickle as pickle
import argparse
import math
import os
import shutil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import data.data_loader as data_utils
import tools.metrics as evaluate
import tools.plot as plot


# from tensorboardX import SummaryWriter


def load_pickle(filename):
    f = open(filename, "rb")
    return pickle.load(f)  # encoding="latin"


def save_pickle(dict_name, file_name):
    with open(file_name, "wb") as fid:
        pickle.dump(dict_name, fid, -1)


class BPR_model(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR_model, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        # prediction_i = (user * item_i).sum(dim=-1)
        prediction_i = torch.matmul(torch.unsqueeze(user, 1), item_i.transpose(1, 2))
        # prediction_j = (user * item_j).sum(dim=-1)
        prediction_j = torch.matmul(torch.unsqueeze(user, 1), item_j.transpose(1, 2))
        # return prediction_i, prediction_j
        return torch.squeeze(prediction_i), torch.squeeze(prediction_j)


class Discriminator(nn.Module):

    def __init__(self, in_dim, out_dim, mid_dim):
        """
        :param in_dim:
        :param out_dim:
        :param mid_dim:
        TODO: specify number of hidden layers.
        """
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(True),
            nn.Linear(mid_dim, out_dim),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Mapping_func(nn.Module):

    def __init__(self, d_model, dropout_rate=0.2):
        super(Mapping_func, self).__init__()
        self.layer1 = nn.Linear(d_model, d_model * 2)
        self.layer2 = nn.Linear(d_model * 2, d_model)
        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inp):
        output = self.layer1(inp)
        # output = self.dropout1(output)
        output = self.relu(output)
        output = self.layer2(output)
        # output = self.dropout2(output)
        return self.layer_norm(output)


class Cross_merge(nn.Module):

    def __init__(self, d_model, dropout_rate=0.2):
        super(Cross_merge, self).__init__()
        self.out_layer = nn.Linear(d_model * 2, d_model)
        self.mid_layer = nn.Linear(d_model * 2, d_model * 2)
        # self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.relu = GELU()

    def forward(self, t_emb, s_emb):
        out = torch.cat([t_emb, s_emb], dim=1)
        out = self.relu(self.mid_layer(out))
        out = self.out_layer(out)
        return out


def train_map(map_func, a_model, b_model, loader, out_dir):
    """
    :param out_dir:
    :param map_func:
    :param map:
    :param a_model:
    :param b_model:
    :param loader:
    :param epoch:
    :return:
    preliminary mapping function training (without mapping function).
    """
    # train mapping function only
    loss = nn.MSELoss()
    opt = optim.Adam(map_func.parameters(), lr=0.006, betas=(0.9, 0.98), eps=1e-09)
    # opt = optim.SGD(map_func.parameters(), lr=0.12, weight_decay=0.001)
    a_model.eval()
    b_model.eval()
    best_val_loss = 100
    for idx in range(200):
        map_func.train()
        for use_b in loader[0]:  # the loader is overlapped user.
            # TODO: in seq rec data loader return two sequence.
            # dummy function get_user_embed (lookup table in real implementation)
            use_b = use_b.cuda()
            used_embed_a = a_model.embed_user(use_b)  # Given user_id, return user embedding.
            used_embed_b = b_model.embed_user(use_b)

            used_embed_a, used_embed_b = used_embed_a.detach(), used_embed_b.detach()
            used_embed_a = map_func(used_embed_a)  # map a to b
            # loss between two embedding. user l2 here.
            loss_ = loss(used_embed_a, used_embed_b)
            loss_.backward()
            opt.step()
        # get valid loss
        map_func.eval()
        val_losses = []
        for use_b in loader[1]:
            use_b = use_b.cuda()
            used_embed_a = a_model.embed_user(use_b)  # Given user_id, return user embedding.
            used_embed_b = b_model.embed_user(use_b)

            used_embed_a, used_embed_b = used_embed_a.detach(), used_embed_b.detach()
            used_embed_a = map_func(used_embed_a)  # map a to b
            # loss between two embedding. user l2 here.
            loss_val = loss(used_embed_a, used_embed_b)
            val_losses.append(loss_val.detach().cpu().numpy())

        plot.plot(out_dir + '/mapping_loss_train', loss_.detach().cpu().numpy())
        # print("\n")
        plot.plot(out_dir + '/mapping_loss_valid', np.mean(val_losses))
        plot.flush()
        plot.tick()
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            # save model.
            torch.save(map_func.state_dict(), out_dir + "/best_map_func")


def train_merge(merge_func, map_func, a_model, b_model, train_loaders,
                val_loaders, test_loaders, steps, lr):
    """
    :return:
    fix recommendation model of target domain, train the function of generating new user embedding,
    a is the target domain. where recommendation is performed.

    """
    # setup
    eval_result = [{}, {}]  # random, freq
    test_result = [{}, {}]
    best_hr_val = 0
    k_val = [1, 5, 10, 20, 30]
    for val in k_val:
        eval_result[0][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
        eval_result[1][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
        test_result[0][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
        test_result[1][str(val)] = {"ht": [], "ndcg": [], "mrr": []}

    metrics_all = ["ht", "ndcg", "mrr"]
    # use different batch size of different data loader to balance these two parts of users.

    # TODO: train the mapping function at the same time.

    optimizer = optim.Adam(merge_func.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-09)

    over_iter = iter(train_loaders[0])  # overlap loader
    only_iter = iter(train_loaders[1])  # domain-only loader
    merge_func.train()
    for step in range(steps):
        try:
            user_2, item_i_2, item_j_2 = next(over_iter)
            user_1, item_i_1, item_j_1 = next(only_iter)
        except:
            over_iter = iter(train_loaders[0])
            only_iter = iter(train_loaders[1])
            user_2, item_i_2, item_j_2 = next(over_iter)
            user_1, item_i_1, item_j_1 = next(only_iter)

        user_2, item_i_2, item_j_2 = user_2.cuda(), item_i_2.cuda(), item_j_2.cuda()
        user_1, item_i_1, item_j_1 = user_1.cuda(), item_i_1.cuda(), item_j_1.cuda()
        # for overlap users.
        user_2_e = merge_func(a_model.embed_user(user_2), b_model.embed_user(user_2))
        # TODO: get embedding, not forward method
        # for domain only users,
        user_1_e = merge_func(a_model.embed_user(user_1), map_func(a_model.embed_user(user_1)))
        merge_func.zero_grad()

        # TODO replace prediction part
        item_i_1 = a_model.embed_item(item_i_1)
        item_j_1 = a_model.embed_item(item_j_1)

        item_i_2 = a_model.embed_item(item_i_2)
        item_j_2 = a_model.embed_item(item_j_2)

        prediction_i_1 = torch.squeeze(torch.matmul(torch.unsqueeze(user_1_e, 1), item_i_1.transpose(1, 2)))
        prediction_j_1 = torch.squeeze(torch.matmul(torch.unsqueeze(user_1_e, 1), item_j_1.transpose(1, 2)))
        loss_1 = - (prediction_i_1 - prediction_j_1).sigmoid().log().sum()
        loss_1.backward()

        prediction_i_2 = torch.squeeze(torch.matmul(torch.unsqueeze(user_2_e, 1), item_i_2.transpose(1, 2)))
        prediction_j_2 = torch.squeeze(torch.matmul(torch.unsqueeze(user_2_e, 1), item_j_2.transpose(1, 2)))
        loss_2 = - (prediction_i_2 - prediction_j_2).sigmoid().log().sum()
        loss_2.backward()

        optimizer.step()
        # if step < 60 or step % 10 == 9:
        eval_r, eval_f = evaluate.metrics_em(merge_func, map_func, a_model, b_model, val_loaders)
        test_r, test_f = evaluate.metrics_em(merge_func, map_func, a_model, b_model, test_loaders)

        for key_tmp in k_val:
            for metric_name in metrics_all:
                eval_result[0][str(key_tmp)][metric_name].extend(eval_r[str(key_tmp)][metric_name])
                eval_result[1][str(key_tmp)][metric_name].extend(eval_f[str(key_tmp)][metric_name])

                test_result[0][str(key_tmp)][metric_name].extend(test_r[str(key_tmp)][metric_name])
                test_result[1][str(key_tmp)][metric_name].extend(test_f[str(key_tmp)][metric_name])

        # elapsed_t = time.time() - start_time
        # print("The time elapse of steps {:03d}".format(step) + " is: " +
        #       time.strftime("%H: %M: %S", time.gmtime(elapsed_t)))
        print("============random")
        print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(eval_result[0]["30"]["ht"][-1],
                                                      eval_result[0]["30"]["ndcg"][-1]))
        print("Test HR: {:.3f}\tNDCG: {:.3f}".format(test_result[0]["30"]["ht"][-1],
                                                     test_result[0]["30"]["ndcg"][-1]))
        print("============frequency")
        print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(eval_result[1]["30"]["ht"][-1],
                                                      eval_result[1]["30"]["ndcg"][-1]))
        print("Test HR: {:.3f}\tNDCG: {:.3f}".format(test_result[1]["30"]["ht"][-1],
                                                     test_result[1]["30"]["ndcg"][-1]))

        if eval_result[1]["30"]["ht"][-1] > best_hr_val:
            best_hr_val, best_ndcg_v = eval_result[1]["30"]["ht"][-1], eval_result[1]["30"]["ndcg"][-1]
            if not os.path.exists(result_):
                os.mkdir(result_)
            torch.save(merge_func.state_dict(), result_ + "/best_merge")
        if step > 200:
            save_pickle([eval_result, test_result], result_file)
    save_pickle([eval_result, test_result], result_file)


parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    type=float,
                    default=0.006,
                    help="learning rate")
parser.add_argument("--lamda",
                    type=float,
                    default=0.001,
                    help="model regularization rate")
parser.add_argument("--batch_size",
                    type=int,
                    default=4096,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=200,
                    help="training epoches")
parser.add_argument("--dataset",
                    type=str,
                    default="movie",
                    help="dataset")
parser.add_argument("--top_k",
                    type=int,
                    default=10,
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=32,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_ng",
                    type=int,
                    default=4,
                    help="sample negative items for training")
parser.add_argument("--test_num_ng",
                    type=int,
                    default=99,
                    help="sample part of negative items for testing")
parser.add_argument("--out",
                    default=True,
                    help="save model or not")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
parser.add_argument("--domain",
                    type=str,
                    default="a",
                    help="gpu card ID")
parser.add_argument("--run",
                    type=int,
                    default=1,
                    help="number of runs")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

# ############################# PREPARE DATASET ##########################

# for mapping function training (user_ids for overlapped users)
dataset = args.dataset
if args.domain == "a":
    domain_idx = 0
else:
    domain_idx = 1
domain_name = dataset.split("_")[domain_idx]
if dataset == "movie_book":
    data_dir = "/data/ceph/seqrec/data/public/Amazon_torch/movie_book"
    n_overlapped = 584
else:
    data_dir = "/data/ceph/seqrec/data/public/Amazon_torch/sport_cloth"
    n_overlapped = 1062

result_dir = "/data/ceph/seqrec/torch/result/mf_%d/emb_map/" % args.factor_num
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
result_dir += dataset + "_%s" % args.run
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
old_model_dir = "/data/ceph/seqrec/torch/result/mf_%d/emb_map" % args.factor_num

train_u, val_u = data_utils.overlap_user_split(n_overlapped, 0.2)
overlap_dataset_train = data_utils.BPRData_EM_over_users(train_u)
overlap_loader_train = data.DataLoader(overlap_dataset_train, batch_size=300, shuffle=True, num_workers=2)

overlap_dataset_val = data_utils.BPRData_EM_over_users(val_u)
overlap_loader_val = data.DataLoader(overlap_dataset_val, batch_size=100, shuffle=False, num_workers=2)

files = [os.path.join(data_dir, dataset + ".pickle"), os.path.join(data_dir, domain_name + "_only.pickle")]
wf = load_pickle(os.path.join(data_dir, "%s_freq.pickle" % domain_name))
result_ = os.path.join(result_dir, domain_name)
if os.path.isdir(result_):
    shutil.rmtree(result_)
os.mkdir(result_)
result_file = result_ + "/result.pickle"

if dataset == "movie_book":
    user_num_a, item_num_a = 4261, 5537
    user_num_b, item_num_b = 42940, 51367
else:
    user_num_a, item_num_a = 9024, 11836
    user_num_b, item_num_b = 46810, 42140
if args.domain == "a":
    user_num_t = user_num_a
    item_num_t = item_num_a
else:
    user_num_t = user_num_b
    item_num_t = item_num_b

# TODO: separate
_, val_feature, test_feature, seqs = data_utils.EM_data_process(files, args.domain)

user_over = np.arange(0, n_overlapped)
user_only = np.arange(n_overlapped, user_num_t)
# train data loader
train_dataset_over = data_utils.BPRData_EM_ct(seqs, item_num_t, user_over, num_ng=1)
train_mat = train_dataset_over.train_mat_gen(user_num_t, item_num_t)
train_dataset_only = data_utils.BPRData_EM_ct(seqs, item_num_t, user_only, num_ng=1)
train_mat = train_dataset_only.train_mat_gen(user_num_t, item_num_t)
# TODO: change the batch_size of the overlap and domainonly data loader to balance the these two parts.
user_ratio = user_num_t / n_overlapped
train_loader_over = data.DataLoader(train_dataset_over, batch_size=int(2048 / user_ratio), shuffle=True, num_workers=4)
train_loader_only = data.DataLoader(train_dataset_only, batch_size=2048, shuffle=True, num_workers=4)
# valid and test data loaders

val_dataset_over = data_utils.BPRData_EM_ce(val_feature, seqs, item_num_t, user_over, num_ng=199, wf=wf)
val_dataset_only = data_utils.BPRData_EM_ce(val_feature, seqs, item_num_t, user_only, num_ng=199, wf=wf)

test_dataset_over = data_utils.BPRData_EM_ce(test_feature, seqs, item_num_t, user_over, num_ng=199, wf=wf)
test_dataset_only = data_utils.BPRData_EM_ce(test_feature, seqs, item_num_t, user_only, num_ng=199, wf=wf)

val_loader_over = data.DataLoader(val_dataset_over, batch_size=100, shuffle=False, num_workers=4)
test_loader_over = data.DataLoader(test_dataset_over, batch_size=100, shuffle=False, num_workers=4)

val_loader_only = data.DataLoader(val_dataset_only, batch_size=100, shuffle=False, num_workers=4)
test_loader_only = data.DataLoader(test_dataset_only, batch_size=100, shuffle=False, num_workers=4)

# ########################## CREATE MODEL #################################
# load recommend model in both domain
model_a = BPR_model(user_num_a, item_num_a, args.factor_num)
model_b = BPR_model(user_num_b, item_num_b, args.factor_num)
# restore.
model_a.load_state_dict(torch.load(os.path.join(old_model_dir, dataset.split("_")[0] + "_%s/best_model" % args.run)))
model_b.load_state_dict(torch.load(os.path.join(old_model_dir, dataset.split("_")[1] + "_%s/best_model" % args.run)))
model_a.cuda()
model_b.cuda()

if args.domain == "a":
    t_model = model_a
    s_model = model_b
else:
    t_model = model_b
    s_model = model_a

test_dataset = data_utils.BPRData_EM_se(test_feature, seqs, item_num_t, num_ng=199, wf=wf)
test_loader_m = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)  # can

eval_result_r, eval_result_f = evaluate.metrics_m(t_model, test_loader_m)
print(eval_result_r, eval_result_f)
# ########################## Train Mapping Function #####################################
map_a2b = Mapping_func(args.factor_num)

map_a2b.cuda()
train_map(map_a2b, t_model, s_model, [overlap_loader_train, overlap_loader_val], result_)

# load the best map function.
map_a2b.load_state_dict(torch.load(result_ + "/best_map_func"))
map_a2b.cuda()
# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamda)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)

del overlap_loader_train
del overlap_loader_val
# writer = SummaryWriter() # for visualization
merge_model = Cross_merge(args.factor_num)
merge_model.cuda()
# ########################## Train merge module #####################################
train_merge(merge_model, map_a2b, t_model, s_model, [train_loader_over, train_loader_only],
            [val_loader_over, val_loader_only], [test_loader_over, test_loader_only], 500, args.lr)

