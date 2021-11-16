import os
import shutil
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import data.data_loader as data_utils
import tools.metrics as evaluate
import tools.plot as plot
from tqdm import tqdm
import _pickle as pickle
import os


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


parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    type=float,
                    default=0.006,
                    help="learning rate")
parser.add_argument("--dataset",
                    type=str,
                    default="sport_cloth",
                    help="which two domain")
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
                    default=300,
                    help="training epoches")
parser.add_argument("--top_k",
                    type=int,
                    default=10,
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=50,
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
parser.add_argument("--run",
                    type=int,
                    default=1,
                    help="number of runs")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

# ############################# PREPARE DATASET ##########################
result_dir_tmp = "/data/ceph/seqrec/torch/result/mf_%d/cross" % args.factor_num
if not os.path.isdir(result_dir_tmp):
    os.mkdir(result_dir_tmp)
if args.dataset == "sport_cloth":
    data_dir = "/data/ceph/seqrec/data/public/Amazon_torch/sport_cloth"

    result_dir = os.path.join(result_dir_tmp, "sport_cloth_%d" % args.run)
    a_domain = "sport"
    b_domain = "cloth"
    result_dir_a = result_dir + "/%s.pickle" % a_domain
    result_dir_b = result_dir + "/%s.pickle" % b_domain
    user_num_a, item_num_a = 9024, 11835 + 1  # sport
    user_num_b, item_num_b = 46810, 42139 + 1  # cloth
else:
    data_dir = "/data/ceph/seqrec/data/public/Amazon_torch/movie_book"
    result_dir = os.path.join(result_dir_tmp, "movie_book_%d" % args.run)
    a_domain = "movie"
    b_domain = "book"
    result_dir_a = result_dir + "/%s.pickle" % a_domain
    result_dir_b = result_dir + "/%s.pickle" % b_domain
    user_num_a, item_num_a = 4261, 5536 + 1  # movie
    user_num_b, item_num_b = 42940, 51366 + 1  # book

if os.path.isdir(result_dir):
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
wf_a = load_pickle(os.path.join(data_dir, a_domain + "_freq.pickle"))
wf_b = load_pickle(os.path.join(data_dir, b_domain + "_freq.pickle"))
print(len(wf_a), len(wf_b), item_num_a, item_num_b)
seqs, val_a_all, val_b_all, test_a_all, test_b_all, a_domain_users, b_domain_users = data_utils.unify_domain_data(
    data_dir, a_domain, b_domain, item_num_a)

train_dataset = data_utils.BPRData_cross_t(seqs, item_num_a + item_num_b, num_g=1)
train_dataset.gen_train_mat(user_num_a + user_num_b, item_num_a + item_num_b)
train_loader_m = data.DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)

# A domain eval and test loader
val_dataset_a = data_utils.BPRData_cross_e(val_a_all, seqs, a_domain_users, item_num_a + item_num_b,
                                           num_ng=199, wf=wf_a, off_set=-item_num_b)
test_dataset_a = data_utils.BPRData_cross_e(test_a_all, seqs, a_domain_users, item_num_a + item_num_b,
                                            num_ng=199, wf=wf_a, off_set=-item_num_b)

val_a_loader = data.DataLoader(val_dataset_a, batch_size=512, shuffle=False, num_workers=4)
test_a_loader = data.DataLoader(test_dataset_a, batch_size=512, shuffle=False, num_workers=4)

# B domain eval and test loader
val_dataset_b = data_utils.BPRData_cross_e(val_b_all, seqs, b_domain_users, item_num_a + item_num_b,
                                           num_ng=199, wf=wf_b, off_set=item_num_a)
test_dataset_b = data_utils.BPRData_cross_e(test_b_all, seqs, b_domain_users, item_num_a + item_num_b,
                                            num_ng=199, wf=wf_b, off_set=item_num_a)

val_b_loader = data.DataLoader(val_dataset_b, batch_size=512, shuffle=False, num_workers=4)
test_b_loader = data.DataLoader(test_dataset_b, batch_size=512, shuffle=False, num_workers=4)

########################### CREATE MODEL #################################
model = BPR_model(user_num_a + user_num_b, item_num_a + item_num_b, args.factor_num)
model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lamda)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09)

# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
count = 0
best_hr_v_a = 0
best_ndcg_v_a = 0
best_epoch_a = 0

best_hr_v_b = 0
best_ndcg_v_b = 0
best_epoch_b = 0

eval_result_a = [{}, {}]  # random, freq
test_result_a = [{}, {}]

eval_result_b = [{}, {}]  # random, freq
test_result_b = [{}, {}]

k_val = [5, 10, 20, 30]
for val in k_val:
    eval_result_a[0][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    eval_result_a[1][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    test_result_a[0][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    test_result_a[1][str(val)] = {"ht": [], "ndcg": [], "mrr": []}

    eval_result_b[0][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    eval_result_b[1][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    test_result_b[0][str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    test_result_b[1][str(val)] = {"ht": [], "ndcg": [], "mrr": []}

metrics_name = ["ht", "ndcg", "mrr"]

# for epoch in tqdm(range(args.epochs)):
for epoch in range(args.epochs):
    model.train()
    start_time = time.time()

    for user, item_i, item_j in train_loader_m:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        model.zero_grad()
        prediction_i, prediction_j = model(user, item_i, item_j)
        loss = - (prediction_i - prediction_j).sigmoid().log().sum()
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        count += 1

    plot.plot(result_dir + '/bpr_loss', loss.cpu().detach().numpy())

    if epoch < 20 or epoch % 10 == 9:
        model.eval()
        eval_result_a_r, eval_result_a_f = evaluate.metrics_m(model, val_a_loader)
        test_result_a_r, test_result_a_f = evaluate.metrics_m(model, test_a_loader)

        eval_result_b_r, eval_result_b_f = evaluate.metrics_m(model, val_b_loader)
        test_result_b_r, test_result_b_f = evaluate.metrics_m(model, test_b_loader)

        for key in k_val:
            for metric in metrics_name:
                eval_result_a[0][str(key)][metric].extend(eval_result_a_r[str(key)][metric])
                eval_result_a[1][str(key)][metric].extend(eval_result_a_f[str(key)][metric])
                test_result_a[0][str(key)][metric].extend(test_result_a_r[str(key)][metric])
                test_result_a[1][str(key)][metric].extend(test_result_a_f[str(key)][metric])

                eval_result_b[0][str(key)][metric].extend(eval_result_b_r[str(key)][metric])
                eval_result_b[1][str(key)][metric].extend(eval_result_b_f[str(key)][metric])
                test_result_b[0][str(key)][metric].extend(test_result_b_r[str(key)][metric])
                test_result_b[1][str(key)][metric].extend(test_result_b_f[str(key)][metric])

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("============random domain %s" % a_domain)
        print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(eval_result_a[0]["30"]["ht"][-1],
                                                      eval_result_a[0]["30"]["ndcg"][-1]))
        print("Test HR: {:.3f}\tNDCG: {:.3f}".format(test_result_a[0]["30"]["ht"][-1],
                                                     test_result_a[0]["30"]["ndcg"][-1]))
        print("============frequency domain %s" % a_domain)
        print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(eval_result_a[1]["30"]["ht"][-1],
                                                      eval_result_a[1]["30"]["ndcg"][-1]))
        print("Test HR: {:.3f}\tNDCG: {:.3f}".format(test_result_a[1]["30"]["ht"][-1],
                                                     test_result_a[1]["30"]["ndcg"][-1]))

        print("============random domain %s" % b_domain)
        print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(eval_result_b[0]["30"]["ht"][-1],
                                                      eval_result_b[0]["30"]["ndcg"][-1]))
        print("Test HR: {:.3f}\tNDCG: {:.3f}".format(test_result_b[0]["30"]["ht"][-1],
                                                     test_result_b[0]["30"]["ndcg"][-1]))
        print("============frequency domain %s" % b_domain)
        print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(eval_result_b[1]["30"]["ht"][-1],
                                                      eval_result_b[1]["30"]["ndcg"][-1]))
        print("Test HR: {:.3f}\tNDCG: {:.3f}".format(test_result_b[1]["30"]["ht"][-1],
                                                     test_result_b[1]["30"]["ndcg"][-1]))

        if eval_result_a[0]["30"]["ht"][-1] > best_hr_v_a:
            best_hr_v_a, best_ndcg_v_a, best_epoch_a = eval_result_a[0]["30"]["ht"][-1], eval_result_a[0]["30"]["ndcg"][
                -1], epoch
            best_hr_t_a, best_ndcg_t_a = test_result_a[0]["30"]["ht"][-1], test_result_a[0]["30"]["ht"][-1]
        if eval_result_b[0]["30"]["ht"][-1] > best_hr_v_b:
            best_hr_v_b, best_ndcg_v_b, best_epoch_b = eval_result_b[0]["30"]["ht"][-1], eval_result_b[0]["30"]["ndcg"][
                -1], epoch
            best_hr_t_b, best_ndcg_t_b = test_result_b[0]["30"]["ht"][-1], test_result_b[0]["30"]["ht"][-1]
    #             if args.out:
    #                 if not os.path.exists(config.model_path):
    #                     os.mkdir(config.model_path)
    #                 torch.save(model, '{}BPR.pt'.format(config.model_path))
        if epoch > 100:
            save_pickle([eval_result_a, test_result_a], result_dir_a)
            save_pickle([eval_result_b, test_result_b], result_dir_b)
    plot.flush('./')
    plot.tick()

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch_a, best_hr_t_a, best_ndcg_t_a))
print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(best_epoch_b, best_hr_t_b, best_ndcg_t_b))
save_pickle([eval_result_a, test_result_a], result_dir_a)
save_pickle([eval_result_b, test_result_b], result_dir_b)
