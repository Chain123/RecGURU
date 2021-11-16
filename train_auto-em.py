# -*- coding: UTF-8 -*-
import _pickle as pickle
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import AutoEnc4Rec as Model
import config_auto4rec as param_c
import data.data_loader as Dataloader
import tools.lossfunctions as lf
import tools.metrics as metrics
import tools.plot as plot


loss_bpr = lf.BPRLoss_sas()


def load_pickle(filename):
    f = open(filename, "rb")
    return pickle.load(f)  # encoding="latin"


def save_pickle(dict_name, file_name):
    with open(file_name, "wb") as fid:
        pickle.dump(dict_name, fid, -1)


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
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.relu = GELU()

    def forward(self, t_emb, s_emb):
        out = torch.cat([t_emb, s_emb], dim=1)
        out = self.relu(self.mid_layer(out))
        out = self.out_layer(out)
        return out


def train_map(map_func, a_model, b_model, loader, out_dir, device_t):
    """
    :param device_t:
    :param out_dir:
    :param map_func:
    :param a_model:
    :param b_model:
    :param loader:
    :return:
    preliminary mapping function training (without mapping function).
    """
    # train mapping function only
    loss = nn.MSELoss()
    opt = optim.Adam(map_func.parameters(), lr=0.008, betas=(0.9, 0.98), eps=1e-09)
    # opt = optim.SGD(map_func.parameters(), lr=0.12, weight_decay=0.001)
    a_model.eval()
    b_model.eval()
    best_val_loss = 100
    for idx in range(200):
        # train on overlap users
        map_func.train()
        for seq_a, seq_b in loader[0]:  # the loader is overlapped user.
            # TODO: in seq rec data loader return two sequence.
            # dummy function get_user_embed (lookup table in real implementation)
            seq_a = seq_a[1].to(device_t)  # decoder in seq.
            seq_b = seq_b[1].to(device_t)
            if torch.cuda.device_count() > 1:
                used_embed_a = a_model.module.get_embedding_sas(seq_a).view(-1, a_model.module.param.d_model)
                # [:, -1, :]  # the recommended embedding.
                used_embed_b = b_model.module.get_embedding_sas(seq_b).view(-1, a_model.module.param.d_model)
            else:
                used_embed_a = a_model.get_embedding_sas(seq_a).view(-1, a_model.param.d_model)
                # [:, -1, :]  # the recommended embedding.
                used_embed_b = b_model.get_embedding_sas(seq_b).view(-1, a_model.param.d_model)

            used_embed_a, used_embed_b = used_embed_a.detach(), used_embed_b.detach()
            used_embed_a = map_func(used_embed_a)  # map a to b
            # loss between two embedding. user l2 here.
            loss_ = loss(used_embed_a, used_embed_b)
            loss_.backward()
            opt.step()  # parameters of the mapping function is updated here.

        # test the mapping function on the rest part of overlap users.
        map_func.eval()
        val_losses = []
        for seq_a, seq_b in loader[1]:
            seq_a = seq_a[1].to(device_t)  # decoder in seq.
            seq_b = seq_b[1].to(device_t)
            if torch.cuda.device_count() > 1:
                used_embed_a = a_model.module.get_embedding_sas(seq_a)[:, -1, :]  # the recommended embedding.
                used_embed_b = b_model.module.get_embedding_sas(seq_b)[:, -1, :]
            else:
                used_embed_a = a_model.get_embedding_sas(seq_a)[:, -1, :]  # the recommended embedding.
                used_embed_b = b_model.get_embedding_sas(seq_b)[:, -1, :]

            used_embed_a, used_embed_b = used_embed_a.detach(), used_embed_b.detach()
            used_embed_a = map_func(used_embed_a)  # map a to b
            # loss between two embedding. user l2 here.
            loss_val = loss(used_embed_a, used_embed_b)
            val_losses.append(loss_val.detach().cpu().numpy())

        plot.plot(out_dir + '/mapping_loss_train', loss_.detach().cpu().numpy())
        # print("\n")
        plot.plot(out_dir + '/mapping_loss_valid', np.mean(val_losses))
        plot.flush(out_dir)
        plot.tick()
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            # save model.
            torch.save(map_func.state_dict(), out_dir + "/best_map_func")


def enhanced_logits(a_model, b_model, merge_func, a_data_over, b_data_over=None, map_func=None):
    for i in range(len(a_data_over)):
        a_data_over[i] = a_data_over[i].cuda()
    if torch.cuda.device_count() > 1:
        used_embed_a = a_model.module.get_embedding_sas(a_data_over[1])  # the recommended embedding.
    else:
        used_embed_a = a_model.get_embedding_sas(a_data_over[1])  # the recommended embedding.
    if b_data_over is not None:
        for i in range(len(b_data_over)):
            b_data_over[i] = b_data_over[i].cuda()
        if torch.cuda.device_count() > 1:
            used_embed_b = b_model.module.get_embedding_sas(b_data_over[1])
        else:
            used_embed_b = b_model.get_embedding_sas(b_data_over[1])
        used_embed_b = used_embed_b.detach()
    elif map_func is not None:
        used_embed_b = map_func(used_embed_a)
    else:
        print("error")
        sys.exit()
    if torch.cuda.device_count() > 1:
        enhanced_user_embed_a = merge_func(used_embed_a.view(-1, a_model.module.param.d_model),
                                           used_embed_b.view(-1, a_model.module.param.d_model))
        enhanced_user_embed_a = enhanced_user_embed_a.view(-1, a_model.module.param.enc_maxlen,
                                                           a_model.module.param.d_model)
        n_items = a_model.module.src_emb(a_data_over[-1]).view(-1, a_model.module.param.enc_maxlen,
                                                               a_model.module.param.num_train_neg,
                                                               a_model.module.param.d_model)
        dec_out = a_model.module.src_emb(a_data_over[2]).view(-1, a_model.module.param.enc_maxlen,
                                                              1, a_model.module.param.d_model)

        enhanced_user_embed_a = enhanced_user_embed_a.view(-1, a_model.module.param.enc_maxlen,
                                                           1, a_model.module.param.d_model)
        mask = a_data_over[1] == a_model.module.param.pad_index
    else:
        enhanced_user_embed_a = merge_func(used_embed_a.view(-1, a_model.param.d_model),
                                           used_embed_b.view(-1, a_model.param.d_model))
        enhanced_user_embed_a = enhanced_user_embed_a.view(-1, a_model.param.enc_maxlen,
                                                           a_model.param.d_model)
        n_items = a_model.src_emb(a_data_over[-1]).view(-1, a_model.param.enc_maxlen,
                                                        a_model.param.num_train_neg,
                                                        a_model.param.d_model)
        dec_out = a_model.src_emb(a_data_over[2]).view(-1, a_model.param.enc_maxlen,
                                                       1, a_model.param.d_model)

        enhanced_user_embed_a = enhanced_user_embed_a.view(-1, a_model.param.enc_maxlen,
                                                           1, a_model.param.d_model)
        mask = a_data_over[1] == a_model.param.pad_index

    p_logits = torch.squeeze(torch.matmul(enhanced_user_embed_a, dec_out.transpose(2, 3)), 2)  # [b, len, 1, 1]
    n_logits = torch.squeeze(torch.matmul(enhanced_user_embed_a, n_items.transpose(2, 3)), 2)  # [b, len, 1, neg]
    mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
    mask.to(device)
    loss = loss_bpr(p_logits, n_logits, mask=mask)
    return loss


def get_scores(a_model, map_func, merge_func, dec_in, target, n_items, param,
               b_model=None, dec_in_b=None, overlap=False):
    if torch.cuda.device_count() > 1:
        used_embed_a = a_model.module.get_embedding_sas(dec_in)[:, -1, :]  # the recommended embedding.
    else:
        used_embed_a = a_model.get_embedding_sas(dec_in)[:, -1, :]  # the recommended embedding.

    if overlap:
        if torch.cuda.device_count() > 1:
            used_embed_b = b_model.module.get_embedding_sas(dec_in_b)[:, -1, :]
        else:
            used_embed_b = b_model.get_embedding_sas(dec_in_b)[:, -1, :]
    else:
        used_embed_b = map_func(used_embed_a)
    enhanced_user_embed_a = merge_func(used_embed_a, used_embed_b)
    if torch.cuda.device_count() > 1:
        n_embeddings = a_model.module.src_emb(n_items).view(-1,
                                                            param.candidate_size,
                                                            param.d_model)
        p_embeddings = a_model.module.src_emb(target).view(-1, param.d_model)
    else:
        n_embeddings = a_model.src_emb(n_items).view(-1,
                                                            param.candidate_size,
                                                            param.d_model)
        p_embeddings = a_model.src_emb(target).view(-1, param.d_model)

    candidate_embeddings = torch.cat([torch.unsqueeze(p_embeddings, 1), n_embeddings], dim=1)  # [B, cand+ 1, d_model]
    scores = torch.matmul(torch.unsqueeze(enhanced_user_embed_a, 1), candidate_embeddings.transpose(1, 2))
    # [B, 1, candidate_size + 1]
    return torch.squeeze(scores)


def evaluate(merge_func, map_func, a_model, val_loaders, de, param, b_model=None):
    k_val = [5, 10, 20, 30]
    merge_func.eval()
    map_func.eval()
    print("evaluation and test")
    rank_eval_f = []
    # rank_eval_r = []
    # rank_test_r = []
    rank_test_f = []
    result_rand = {}
    result_freq = {}
    for val in k_val:
        result_rand[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                                 "ht_test": [], "ndcg_test": [], "mrr_test": []}
        result_freq[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                                 "ht_test": [], "ndcg_test": [], "mrr_test": []}

    eval_iterator = iter(val_loaders[1])  # non-overlap
    # for all the non-overlap users.
    while True:
        try:
            eval_data, test_data, n_items_f, n_items_r = next(eval_iterator)
        except StopIteration:
            break
        n_items_f = n_items_f.to(de)

        eval_dec_in, eval_target = eval_data[1].to(de), eval_data[2].to(de)
        eval_score_f = get_scores(a_model, map_func, merge_func, eval_dec_in, eval_target,
                                  n_items_f, param)
        test_dec_in, test_target = test_data[1].to(de), test_data[2].to(de)
        test_score_f = get_scores(a_model, map_func, merge_func, test_dec_in, test_target,
                                  n_items_f, param)
        rank_eval_f.extend(torch.argsort(torch.argsort(-eval_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_test_f.extend(torch.argsort(torch.argsort(-test_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())

    eval_iterator = iter(val_loaders[0])  # overlap data
    while True:
        try:
            a_data, b_data = next(eval_iterator)
        except StopIteration:
            break
        n_items_f = a_data[-1].to(de)

        eval_dec_in, eval_target = a_data[1].to(de), a_data[2].to(de)
        b_in = b_data[1].to(de)
        eval_score_f = get_scores(a_model, map_func, merge_func, eval_dec_in, eval_target,
                                  n_items_f, param, b_model=b_model, dec_in_b=b_in, overlap=True)

        test_dec_in, test_target = a_data[4].to(de), a_data[5].to(de)
        b_in = b_data[-1].to(de)
        test_score_f = get_scores(a_model, map_func, merge_func, test_dec_in, test_target,
                                  n_items_f, param, b_model=b_model, dec_in_b=b_in, overlap=True)

        rank_eval_f.extend(torch.argsort(torch.argsort(-eval_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_test_f.extend(torch.argsort(torch.argsort(-test_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())

    # TODO metrics
    for k in k_val:
        result_freq[str(k)]["ht_eval"].append(metrics.hit_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["ht_test"].append(metrics.hit_at_k_batch(rank_test_f, k))
        result_freq[str(k)]["ndcg_eval"].append(metrics.NDCG_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["ndcg_test"].append(metrics.NDCG_at_k_batch(rank_test_f, k))
        result_freq[str(k)]["mrr_eval"].append(metrics.mrr_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["mrr_test"].append(metrics.mrr_at_k_batch(rank_test_f, k))
    return result_freq


def train_merge(merge_func, map_func, a_model, b_model, train_loaders,
                val_loader, steps, lr, result_file_in):
    """
    :return:
    fix recommendation model of target domain, train the function of generating new user embedding,
    a is the target domain. where recommendation is performed.

    """
    result = {}
    best_hr_val = 0
    k_val = [5, 10, 20, 30]
    for val in k_val:
        result[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                            "ht_test": [], "ndcg_test": [], "mrr_test": []}
    metrics_all = list(result[str(k_val[0])].keys())

    optimizer = optim.Adam(merge_func.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-09)

    over_iter = iter(train_loaders[0])  # overlap loader
    only_iter = iter(train_loaders[1])  # domain-only loader
    merge_func.train()
    map_func.train()
    for step in range(steps):
        try:
            a_data_over, b_data_over = next(over_iter)
            data_only = next(only_iter)
            # b_data_only = next(only_iter_b)
        except StopIteration:
            over_iter = iter(train_loaders[0])  # overlap loader
            only_iter = iter(train_loaders[1])  # domain-only loader
            # only_iter_b = iter(train_loaders[2])

            a_data_over, b_data_over = next(over_iter)
            data_only = next(only_iter)
            # b_data_only = next(only_iter_b)
        merge_func.zero_grad()
        map_func.zero_grad()
        # for overlap users.
        loss_over = enhanced_logits(a_model, b_model, merge_func, a_data_over, b_data_over=b_data_over, map_func=None)
        # for domain only users,
        loss_only = enhanced_logits(a_model, b_model, merge_func, data_only, b_data_over=None, map_func=map_func)

        # domain only prediction.
        loss_over.backward()
        loss_only.backward()
        optimizer.step()

        if step < 20 or step % 10 == 9:
            if torch.cuda.device_count() > 1:
                result_tmp = evaluate(merge_func, map_func, a_model,
                                      val_loader, device, a_model.module.param, b_model)
            else:
                result_tmp = evaluate(merge_func, map_func, a_model,
                                      val_loader, device, a_model.param, b_model)
            for key_tmp in k_val:
                key_tmp = str(key_tmp)
                for metric_name in metrics_all:
                    result[key_tmp][metric_name].extend(result_tmp[key_tmp][metric_name])
                    # result[1][key_tmp][metric_name].extend(result_tmp[key_tmp][metric_name])
            # print("============random")
            # print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(result_tmp["30"]["ht_eval"][-1],
            #                                               result_tmp["30"]["ndcg_eval"][-1]))
            # print("Test HR: {:.3f}\tNDCG: {:.3f}".format(result_tmp["30"]["ht_test"][-1],
            #                                              result_tmp["30"]["ndcg_test"][-1]))
            print("============frequency")
            print("Valid HR: {:.3f}\tNDCG: {:.3f}".format(result_tmp["30"]["ht_eval"][-1],
                                                          result_tmp["30"]["ndcg_eval"][-1]))
            print("Test HR: {:.3f}\tNDCG: {:.3f}".format(result_tmp["30"]["ht_test"][-1],
                                                         result_tmp["30"]["ndcg_test"][-1]))

            if result_tmp["30"]["ht_eval"][-1] > best_hr_val:
                best_hr_val, best_ndcg_v = result_tmp["30"]["ht_eval"][-1], result_tmp["30"]["ndcg_eval"][-1]
                if not os.path.exists(map_result_):
                    os.mkdir(map_result_)
                torch.save(merge_func.state_dict(), map_result_ + "/best_merge")
            if step > 200:
                save_pickle(result, result_file_in)
    save_pickle(result, result_file_in)


def str2bool_par(val):
    if val == "True":
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.01,
                        help="learning rate")
    parser.add_argument("--date",
                        type=str,
                        default="sas_org",
                        help="labeling")
    parser.add_argument("--d_model",
                        type=int,
                        default=32,
                        help="predictive factors numbers in the model")
    parser.add_argument("--n_head",
                        type=int,
                        default=1,
                        help="multi-headed number of head")
    parser.add_argument("--d_ff",
                        type=int,
                        default=512,
                        help="feed-forward dim")
    parser.add_argument("--n_negs",
                        type=int,
                        default=30,
                        help="number of negs in sampled softmax")
    parser.add_argument("--decoder_neg",
                        type=bool,
                        default=True,
                        help="whether to use negative sampling while training enc-dec")
    # decoder_neg = True  # whether to use negative sampling while training enc-dec
    parser.add_argument("--fixed_enc",
                        type=bool,
                        default=True,
                        help="whether the shared_enc can be updated by rs loss only")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024,
                        help="batch size for training")
    parser.add_argument("--batch_size_val",
                        type=int,
                        default=128,
                        help="batch size for training")
    parser.add_argument("--target_domain",
                        type=str,
                        default="a",
                        help="domain")
    parser.add_argument("--dataset_pick",
                        type=int,
                        default=1,
                        help="1 for movie_book, 2 for sport_cloth, 2 for wesee-txvideo")
    parser.add_argument("--run",
                        type=int,
                        default=1,
                        help="number of runs")
    parser.add_argument("--n_gpu",
                        type=int,
                        default=1,
                        help="number of gpus")
    parser.add_argument("--result_path",
                        type=str,
                        default="/data/ceph/seqrec/torch/result/gur_s/non_shared/",
                        help="number of runs")
    parser.add_argument("--sas",
                        type=str,
                        default="False",
                        help="whether to use sas")
    parser.add_argument("--cross",
                        type=str,
                        default="False",
                        help="whether to use sas")
    parser.add_argument("--share_dec",
                        type=str,
                        default="False",
                        help="whether to use sas")
    parser.add_argument("--fix_enc",
                        type=str,
                        default="True",
                        help="whether to use sas")
    parser.add_argument("--org_model_path",
                        type=str,
                        default="pre-trained recommendation model path",
                        help="whether to use sas")

    args = parser.parse_args()
    target_domain = args.target_domain

    args.target_domain = "a"
    param_a = param_c.get_param(args)

    args.target_domain = "b"
    param_b = param_c.get_param(args)

    if target_domain == "a":
        domain_name = param_a.domain_name
    else:
        domain_name = param_b.domain_name

    # ############################# PREPARE DATASET ##########################
    overlap_file = os.path.join(param_a.data_path, "_".join([param_a.domain_name, param_b.domain_name]) + ".pickle")
    overlap_loader_train, overlap_loader_val = Dataloader.dataloader_gen_over_em(overlap_file,
                                                                                 param_a, n_w=2,
                                                                                 ratio=0.2,
                                                                                 target=target_domain)
    # overlap_loader_train = None
    # overlap_loader_val = None
    map_result_ = args.result_path
    recom_train_files = [overlap_file,
                         os.path.join(param_a.data_path, "%s_only.pickle" % domain_name)]
    # test_files = [os.path.join(param_a.data_path, "%s_only.pickle" % domain_name)]

    if target_domain == "a":
        train_loader_over, train_loader_only = Dataloader.dataloader_em_train(recom_train_files,
                                                                              param_a, n_w=2,
                                                                              wf=None,
                                                                              target="a")
    else:
        train_loader_over, train_loader_only = Dataloader.dataloader_em_train(recom_train_files,
                                                                              param_b,
                                                                              n_w=2,
                                                                              wf=None,
                                                                              target="b")

    # val_loader_over, val_loader_only = None, None
    # test_loader_over, test_loader_only = None, None
    freq_file = [os.path.join(param_a.data_path, name) for name in os.listdir(param_a.data_path) if
                 domain_name in name and "freq" in name]

    if len(freq_file) > 0:
        item_fre = Dataloader.load_pickle(freq_file[0])
    else:
        item_fre = None
    val_loader_over, val_loader_only = Dataloader.dataloader_gen_em_val(recom_train_files, param_a,
                                                                        wf=item_fre, target=target_domain)
    # ########################## CREATE MODEL #################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a = Model.MyRec(device,
                          param_a,
                          wf=None,
                          dec_rec=str2bool_par(args.share_dec),
                          fix_enc=str2bool_par(args.fix_enc),
                          sas=str2bool_par(args.sas),
                          pos_train=False).to(torch.float32).to(device)

    model_b = Model.MyRec(device,
                          param_b,
                          wf=None,
                          dec_rec=str2bool_par(args.share_dec),
                          fix_enc=str2bool_par(args.fix_enc),
                          sas=str2bool_par(args.sas),
                          pos_train=False).to(torch.float32)

    # restore.
    model_a.load_state_dict(torch.load(os.path.join(args.org_model_path,
                                                    "%s_%d_%d_mg" % (
                                                        param_a.domain_name, args.d_model, args.run),
                                                    "model")))
    model_b.load_state_dict(torch.load(os.path.join(args.org_model_path,
                                                    "%s_%d_%d_mg" % (
                                                        param_b.domain_name, args.d_model, args.run),
                                                    "model")))
    if torch.cuda.device_count() > 1:
        model_a = nn.DataParallel(model_a)
        model_b = nn.DataParallel(model_b)

    if target_domain == "a":
        t_model = model_a
        s_model = model_b
    else:
        t_model = model_b
        s_model = model_a
    # ########################## Train Mapping Function #####################################
    map_a2b = Mapping_func(args.d_model)
    map_a2b.cuda()
    train_map(map_a2b, t_model, s_model, [overlap_loader_train, overlap_loader_val], map_result_, device)
    # load the best map function.
    map_a2b.load_state_dict(torch.load(map_result_ + "/best_map_func"))
    map_a2b.cuda()

    merge_model = Cross_merge(args.d_model)
    merge_model.cuda()
    # ########################## Train merge module #####################################
    if target_domain == "a":
        name = param_a.domain_name + "_%d_%d" % (args.d_model, args.run)
    else:
        name = param_b.domain_name + "_%d_%d" % (args.d_model, args.run)
    result_file = os.path.join(args.result_path, "result_%s.pickle" % name)
    train_merge(merge_model, map_a2b, t_model, s_model, [train_loader_over, train_loader_only],
                [val_loader_over, val_loader_only], 450, args.lr, result_file)
