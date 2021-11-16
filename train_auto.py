# -*- coding: UTF-8 -*-
"""
AutoEnc4Rec model for sequential recommendation:
Encoder -> embedding -> sequence decoder.
"""
import sys

import Transformer.transformer as all_module
from tools.metrics import *
import torch
import config_auto4rec as param_c
import data.data_loader as Dataloader
from tqdm import tqdm
import os
import torch.nn as nn
import AutoEnc4Rec as Model
import torch.optim as optim
import tools.lossfunctions as lf
import tools.plot as plot
import tools.metrics as metrics
import argparse
from tqdm import tqdm

loss_ce = nn.CrossEntropyLoss(reduction="none")
loss_s_ce = lf.SampledCrossEntropyLoss(reduction="none")
loss_bpr = lf.BPRLoss_sas()


def loss_ae(model, enc_in, dec_in, dec_out, n_items, neg_sample, bs, sl, param_config, mask):
    """
    :param model:  Autoencoder model
    :param enc_in:
    :param dec_in:
    :param dec_out:
    :param n_items:
    :param neg_sample:
    :param bs:
    :param sl:
    :param param_config:
    :param mask:
    :return:
    """
    logits = model(enc_in, dec_in, dec_out, n_items, recon=True)
    if neg_sample:
        # negative sample use fake label
        label = torch.zeros(bs * sl).long().to(device)
        num_class = param_config.n_negs + 1
    else:
        label = dec_out.view(-1, 1).long()
        num_class = param_config.vocab_size + 1
    loss_m = loss_s_ce(logits, label, num_class, mask=mask)
    # loss = loss_ce(logits.view(-1, num_class), label)
    # loss_m = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
    return loss_m


def loss_bpr_func(model_train, enc_in, dec_in, dec_out, n_items, mask):
    """
    :param mask:         input padding mask
    :param model_train:  wrapper model.
    :param enc_in:
    :param dec_in:
    :param dec_out:
    :param n_items:
    :return:
    """
    p_logits, n_logits = model_train(enc_in, dec_in, dec_out, n_items)
    loss = loss_bpr(p_logits, n_logits, mask=mask)
    return loss


def get_next_batch(data_batch):
    """
    Batch for training autoencoder.
    :param data_batch:
    :return:
    """
    seqs, n_items, val, test = data_batch[0], data_batch[1], data_batch[2], data_batch[3]
    n_items, val, test = n_items.to(device), val.to(device), test.to(device)
    enc_in, dec_in, dec_out = seqs[0], seqs[1], seqs[2]
    bs, sl = dec_out.shape[0], dec_out.shape[1]
    enc_in, dec_in, dec_out = enc_in.to(device), dec_in.to(device), dec_out.to(device)
    return enc_in, dec_in, dec_out, n_items, val, test, bs, sl


def train(model_train, opt, steps, data, param_config, device_i, neg_sample=True,
          loss_type="s_soft", opt_type="org", eval_loader=None, k_val=None, sas=False):
    if k_val is None:
        k_val = [5, 10, 20, 30]
    model_train.train()
    result = [{}, {}]
    for val in k_val:
        result[0][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
        result[1][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
    k_val_str = list(result[0].keys())
    metrics_name = list(result[0][k_val_str[0]].keys())
    orig_epo = 0

    for epoch in range(500):  # steps: number of epochs,
        pbar = tqdm(enumerate(data[0]))
        pbar.set_description(f'[Train epoch {epoch}]')
        loss_epoch = 0
        n_batch = 0
        for ind, data_batch in pbar:
            enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(data_batch)
            # mask
            mask = dec_in == param_config.pad_index
            mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
            mask.to(device_i)
            opt.zero_grad()
            # TODO: implement different losses for different tasks.
            if loss_type == "s_soft":
                # selected softmax: reconstruction loss for auto encoder training.
                loss = loss_ae(model_train, enc_in, dec_in, dec_out, n_items, neg_sample, bs, sl, param_config, mask)
            elif loss_type == "bpr":
                # BPR loss for training the recommendation loss
                loss = loss_bpr_func(model_train, enc_in, dec_in, dec_out, n_items, mask)
            else:
                print('Wrong loss config')
                sys.exit()
            # back propagation
            loss.backward()
            # update
            if opt_type == "org":
                opt.step()  # normal optimizer (Adam)
            else:
                opt.step_and_update_lr()
            torch.cuda.empty_cache()
            loss_epoch += loss.item()
            n_batch = ind
            pbar.set_postfix(loss=loss.item())
        loss_epoch /= n_batch

        # validation
        if loss_type == "s_soft":
            print(f"Reconstruction loss after {epoch} epochs: {loss_epoch}")
            plot.plot(param_config.result_path + '/reconstruct_loss_%s' %
                      param_config.date,
                      loss_epoch)
        else:
            print(f"BPR loss after {epoch} epochs: {loss_epoch}")
            plot.plot(param_config.result_path + '/bpr_loss_%s' %
                      param_config.date,
                      loss_epoch)
            # evaluate
            result_tmp = evaluation(model_train, eval_loader, device_i, param_config, sas=sas)
            # print("hello ")
            print("Freq eval HT@10 %f, test HT@10 %f" % (result_tmp[0]["10"]["ht_eval"][0],
                                                         result_tmp[0]["10"]["ht_test"][0]))
            print("Random eval HT@10 %f, test HT@10 %f" % (result_tmp[1]["10"]["ht_eval"][0],
                                                           result_tmp[1]["10"]["ht_test"][0]))
            for key in k_val_str:
                for metric in metrics_name:
                    result[0][key][metric].extend(result_tmp[0][key][metric])
                    result[1][key][metric].extend(result_tmp[1][key][metric])
            Dataloader.save_pickle(result, param_config.result_path + "/result_%s.pickle" % param_config.date)
            model_train.train()
        plot.flush(param_config.result_path)
        plot.tick()


def get_scores(model, enc_in, dec_in, target, n_items, param, sas=False):
    if sas:
        if torch.cuda.device_count() > 1:
            dec_embedding = model.module.get_embedding_sas(dec_in)
            n_embeddings = model.module.src_emb(n_items).view(-1,
                                                              param.candidate_size,
                                                              param.d_model)
            p_embeddings = model.module.src_emb(target).view(-1, param.d_model)
        else:
            dec_embedding = model.get_embedding_sas(dec_in)
            n_embeddings = model.src_emb(n_items).view(-1,
                                                       param.candidate_size,
                                                       param.d_model)
            p_embeddings = model.src_emb(target).view(-1, param.d_model)
        dec_embedding = dec_embedding[:, -1, :]  # [B, d_model]  pred_embedding
    else:
        if torch.cuda.device_count() > 1:
            dec_embedding = model.module.get_embedding(enc_in, dec_in)
            n_embeddings = model.module.AutoEnc.src_emb(n_items).view(-1,
                                                                      param.candidate_size,
                                                                      param.d_model)
            p_embeddings = model.module.AutoEnc.src_emb(target).view(-1,
                                                                     param.d_model)
        else:
            dec_embedding = model.get_embedding(enc_in, dec_in)
            n_embeddings = model.AutoEnc.src_emb(n_items).view(-1,
                                                               param.candidate_size,
                                                               param.d_model)
            p_embeddings = model.AutoEnc.src_emb(target).view(-1,
                                                              param.d_model)
        dec_embedding = dec_embedding[:, -1, :]  # [B, d_model]  pred_embedding

    candidate_embeddings = torch.cat([torch.unsqueeze(p_embeddings, 1), n_embeddings], dim=1)  # [B, cand+ 1, d_model]
    scores = torch.matmul(torch.unsqueeze(dec_embedding, 1), candidate_embeddings.transpose(1, 2))
    # [B, 1, candidate_size + 1]
    return torch.squeeze(scores)


def evaluation(model_train, data_loader, de, param_config, k_val=None, sas=False):
    if k_val is None:
        k_val = [5, 10, 20, 30]
    model_train.eval()
    print("evaluation and test")
    rank_eval_f = []
    rank_eval_r = []
    rank_test_f = []
    rank_test_r = []
    result_rand = {}
    result_freq = {}
    for val in k_val:
        result_rand[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                                 "ht_test": [], "ndcg_test": [], "mrr_test": []}
        result_freq[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                                 "ht_test": [], "ndcg_test": [], "mrr_test": []}

    pbar = tqdm(enumerate(data_loader))
    pbar.set_description(f'[Evaluation and testing: ]')
    for ind, data_batch in pbar:
        eval_data, test_data, n_items_f, n_items_r = data_batch[0], data_batch[1], data_batch[2], data_batch[3]
        # eval
        n_items_f = n_items_f.to(de)
        n_items_r = n_items_r.to(de)

        eval_enc_in, eval_dec_in, eval_target = eval_data[0].to(de), eval_data[1].to(de), eval_data[2].to(de)
        eval_score_f = get_scores(model_train, eval_enc_in, eval_dec_in, eval_target, n_items_f, param_config, sas)
        eval_score_r = get_scores(model_train, eval_enc_in, eval_dec_in, eval_target, n_items_r, param_config, sas)
        # test
        test_enc_in, test_dec_in, test_target = test_data[0].to(de), test_data[1].to(de), test_data[2].to(de)
        test_score_f = get_scores(model_train, test_enc_in, test_dec_in, test_target, n_items_f, param_config, sas)
        test_score_r = get_scores(model_train, test_enc_in, test_dec_in, test_target, n_items_r, param_config, sas)
        rank_eval_f.extend(torch.argsort(torch.argsort(-eval_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_eval_r.extend(torch.argsort(torch.argsort(-eval_score_r, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_test_f.extend(torch.argsort(torch.argsort(-test_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_test_r.extend(torch.argsort(torch.argsort(-test_score_r, dim=1), dim=1)[:, 0].cpu().detach().numpy())
    for k in k_val:
        result_rand[str(k)]["ht_eval"].append(metrics.hit_at_k_batch(rank_eval_r, k))
        result_rand[str(k)]["ht_test"].append(metrics.hit_at_k_batch(rank_test_r, k))
        result_rand[str(k)]["ndcg_eval"].append(metrics.NDCG_at_k_batch(rank_eval_r, k))
        result_rand[str(k)]["ndcg_test"].append(metrics.NDCG_at_k_batch(rank_test_r, k))
        result_rand[str(k)]["mrr_eval"].append(metrics.mrr_at_k_batch(rank_eval_r, k))
        result_rand[str(k)]["mrr_test"].append(metrics.mrr_at_k_batch(rank_test_r, k))

        result_freq[str(k)]["ht_eval"].append(metrics.hit_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["ht_test"].append(metrics.hit_at_k_batch(rank_test_f, k))
        result_freq[str(k)]["ndcg_eval"].append(metrics.NDCG_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["ndcg_test"].append(metrics.NDCG_at_k_batch(rank_test_f, k))
        result_freq[str(k)]["mrr_eval"].append(metrics.mrr_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["mrr_test"].append(metrics.mrr_at_k_batch(rank_test_f, k))

    return [result_freq, result_rand]


def train_ae(model_train, opt, steps, data, param_config, device_i, neg_sample=True):
    """
    :param device_i: 
    :param param_config:
    :param opt:
    :param model_train:
    :param steps:
    :param data:
    :param neg_sample:
    :return:
    """
    model_train.train()
    dataloader_iterator = iter(data)
    for i in tqdm(range(steps)):
        try:
            seqs, n_items, val, test = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(data)
            seqs, n_items, val, test = next(dataloader_iterator)
        # TODO put his in data loader to improve efficiency.
        n_items, val, test = n_items.to(device_i), val.to(device_i), test.to(device_i)
        opt.zero_grad()
        enc_in, dec_in, dec_out = seqs[0], seqs[1], seqs[2]
        bs, sl = dec_out.shape[0], dec_out.shape[1]
        enc_in, dec_in, dec_out = enc_in.to(device_i), dec_in.to(device_i), dec_out.to(device_i)
        logits = model_train(enc_in, dec_in, dec_out, n_items)
        if neg_sample:
            # negative sample use fake label
            label = torch.zeros(bs * sl).long().to(device_i)
            num_class = param_config.n_negs + 1
        else:
            # softmax on whole vocab, use true label
            label = dec_out.view(-1, 1).long()
            num_class = param_config.vocab_size
        mask = dec_in == param_config.pad_index
        mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
        mask.to(device_i)
        # cross-entropy loss
        loss = loss_ce(logits.view(-1, num_class), label)
        loss_m = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
        # back propagation
        loss_m.backward()
        # update
        # opt.step()  # normal optimizer (Adam)
        opt.step_and_update_lr()
        if device_i == "cpu":
            loss_epoch = loss_m.detach().numpy()
        else:
            loss_epoch = loss_m.cpu().detach().numpy()
        # eval step.
        if i % 50 == 49:
            print("reconstruction loss after %d batch" % i, loss_epoch)


def print_param(model, mul=False):
    if mul:
        parameters = model.module.named_parameters()
    else:
        parameters = model.named_parameters()
    for n, par in parameters:
        if par.requires_grad:
            print(n)
            print(par.is_cuda)


def str2bool_par(val):
    if val == "True":
        return True
    else:
        return False


def main(args_t, device_t, train_loader_ae, train_loader_re, eval_loader,
         loader_re_f, item_freq=None, sas_=False, shared=None, fix_enc=None):
    # devices = list(range(n_g))
    sas_ = str2bool_par(sas_)  # whether to build the sas baseline model.
    if sas_:
        shared = False
        fix_enc = False
    else:
        shared = str2bool_par(shared)
        fix_enc = str2bool_par(fix_enc)
    print(sas_)

    MyModel = Model.MyRec(device_t,
                          args_t,
                          wf=item_freq,
                          dec_rec=shared,
                          fix_enc=fix_enc,
                          sas=sas_,
                          pos_train=False).to(torch.float32).to(device_t)
    MyModel = MyModel.to(device_t)
    # print out model parameters:
    # print_param(MyModel)
    if torch.cuda.device_count() > 1:
        MyModel = nn.DataParallel(MyModel)
    # pre-train the AutoEncoder
    if not sas_:
        opt = all_module.ScheduledOptim(
            optim.Adam(MyModel.parameters(), betas=(0.9, 0.99), eps=1e-09),
            1.0, args_t.d_model, args_t.n_warmup_steps)
        train(MyModel, opt, args_t.training_steps, [train_loader_ae, train_loader_ae], args_t, device_t,
              neg_sample=True, loss_type="s_soft", opt_type="schedule")
    # fine-tune on recommendation task.
    if not sas_:
        lr = args_t.lr_rs
    else:
        lr = args_t.lr_rs
    opt_rec = optim.Adam(MyModel.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-09)
    train(MyModel, opt_rec, int(args_t.training_steps_tune), [train_loader_re, loader_re_f],
          args_t, device_t, neg_sample=True, loss_type="bpr", eval_loader=eval_loader, sas=sas_)
    if torch.cuda.device_count() > 1:
        torch.save(MyModel.module.state_dict(), os.path.join(args_t.result_path, "model/model"))
    else:
        torch.save(MyModel.state_dict(), os.path.join(args_t.result_path, "model/model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--date", type=str, default="sas_org", help="labeling")
    parser.add_argument("--d_model", type=int, default=32, help="dimensionality of latent space")
    parser.add_argument("--n_head", type=int, default=1, help="multi-headed number of head")
    parser.add_argument("--d_ff", type=int, default=512, help="feed-forward dim")
    parser.add_argument("--n_negs", type=int, default=30, help="number of negs in sampled softmax")
    parser.add_argument("--decoder_neg", type=bool, default=True,
                        help="whether to use negative sampling while training enc-dec")
    parser.add_argument("--fixed_enc", type=bool, default=True,
                        help="whether the shared_enc can be updated by rs loss only")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=128, help="batch size for training")
    parser.add_argument("--target_domain", type=str, default="a", help="target domain")
    parser.add_argument("--dataset_pick", type=int, default=1,
                        help="1 for movie_book, 2 for sport_cloth, 2 for wesee-txvideo")
    parser.add_argument("--run", type=int, default=1, help="repeat round of experiments")
    # parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus")
    parser.add_argument("--result_path", type=str,
                        default="/data/ceph/seqrec/data_guru/result_test",
                        help="number of runs")
    parser.add_argument("--sas", type=str, default="False", help="whether to use sas")
    parser.add_argument("--cross", type=str, default="False", help="whether to use sas")
    parser.add_argument("--share_dec", type=str, default="False", help="whether to use sas")
    parser.add_argument("--fix_enc", type=str, default="True", help="whether to use sas")
    args = parser.parse_args()
    param = param_c.get_param(args)

    # config data and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if not os.path.isdir(param.result_path):
        os.mkdir(param.result_path)

    all_files = [os.path.join(param.data_path, name) for name in os.listdir(param.data_path) if
                 param.domain_name in name and "_" not in name]
    freq_file = [os.path.join(param.data_path, name) for name in os.listdir(param.data_path) if
                 param.domain_name in name and "freq" in name]
    if len(freq_file) > 0:
        item_fre = Dataloader.load_pickle(freq_file[0])
    else:
        item_fre = None

    train_loader_AE = Dataloader.dataloader_gen(all_files, param, param.n_negs,
                                                train=True, seq_len=None, wf=None,
                                                domain=param.target_domain)
    train_loader_RE = Dataloader.dataloader_gen(all_files, param, param.n_bpr_neg,
                                                train=True, seq_len=None, wf=None,
                                                rec=True, domain=param.target_domain)

    train_loader_re_f = Dataloader.dataloader_gen(all_files, param, param.n_bpr_neg,
                                                  train=True, seq_len=None, wf=item_fre,
                                                  rec=True, domain=param.target_domain)

    train_loader_re_test = Dataloader.dataloader_gen(all_files, param, param.n_bpr_neg,
                                                     train=False, seq_len=None, wf=item_fre,
                                                     domain=param.target_domain)

    main(param, device, train_loader_AE, train_loader_RE, train_loader_re_test, train_loader_re_f,
         item_freq=item_fre, sas_=args.sas, shared=args.share_dec, fix_enc=args.fix_enc)
