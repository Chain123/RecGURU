# -*- coding: UTF-8 -*-
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import data.data_loader as Dataloader
import tools.metrics as metrics
import tools.plot as plot
from tools.utils import get_next_batch, loss_bpr_func, loss_ae

sys.path.append(os.getcwd())
matplotlib.use('Agg')
torch.manual_seed(1)
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 5  # How many critic iterations per generator iteration

date = "1209"
l2_loss = nn.MSELoss()


class l2_constraint(nn.Module):

    def __init__(self):
        super(l2_constraint, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward_2(self, a_embed, b_embed):
        return self.l2_loss(a_embed, b_embed)


def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, device):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    # alpha = alpha.cuda() if use_cuda else alpha
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # if use_cuda:
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def get_scores(model, enc_in, dec_in, target, n_items, param, sas=False, domain="a", device=None):
    if sas:
        n_embeddings = None
        p_embeddings = None
        dec_embedding = None
    else:
        enc_mask = get_pad_mask(dec_in, param.pad_index, device)
        if torch.cuda.device_count() > 1:
            dec_embedding = model.module.recommend_forward(enc_in, dec_in, domain, enc_mask)
            dec_embedding = dec_embedding[:, -1, :]  # [B, d_model]
            if domain == "a":
                n_embeddings = model.module.src_emb_a(n_items).view(-1, param.candidate_size, param.d_model)
                p_embeddings = model.module.src_emb_a(target).view(-1, param.d_model)
            else:
                n_embeddings = model.module.src_emb_b(n_items).view(-1, param.candidate_size, param.d_model)
                p_embeddings = model.module.src_emb_b(target).view(-1, param.d_model)
        else:
            dec_embedding = model.recommend_forward(enc_in, dec_in, domain, enc_mask)
            dec_embedding = dec_embedding[:, -1, :]  # [B, d_model]
            if domain == "a":
                n_embeddings = model.src_emb_a(n_items).view(-1, param.candidate_size, param.d_model)
                p_embeddings = model.src_emb_a(target).view(-1, param.d_model)
            else:
                n_embeddings = model.src_emb_b(n_items).view(-1, param.candidate_size, param.d_model)
                p_embeddings = model.src_emb_b(target).view(-1, param.d_model)

    candidate_embeddings = torch.cat([torch.unsqueeze(p_embeddings, 1), n_embeddings], dim=1)  # [B, cand + 1, d_model]
    scores = torch.matmul(torch.unsqueeze(dec_embedding, 1), candidate_embeddings.transpose(1, 2))
    # [B, 1, candidate_size + 1]
    return torch.squeeze(scores)


def evaluation_2(model_train, data_loader, de, param, k_val=None, sas=False, domain="a"):
    if k_val is None:
        k_val = [1, 5, 10, 20, 30]
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

    eval_iterator = iter(data_loader)
    for _ in tqdm(range(param.eval_steps)):
        try:
            eval_data, test_data, n_items_f, n_items_r = next(eval_iterator)
        except StopIteration:
            eval_iterator = iter(data_loader)
            eval_data, test_data, n_items_f, n_items_r = next(eval_iterator)
        # eval
        # print(type(eval_data))
        n_items_f = n_items_f.to(de)
        n_items_r = n_items_r.to(de)

        eval_enc_in, eval_dec_in, eval_target = eval_data[0].to(de), eval_data[1].to(de), eval_data[2].to(de)
        eval_score_f = get_scores(model_train, eval_enc_in, eval_dec_in, eval_target,
                                  n_items_f, param, sas, domain, de)
        eval_score_r = get_scores(model_train, eval_enc_in, eval_dec_in, eval_target,
                                  n_items_r, param, sas, domain, de)
        # test
        test_enc_in, test_dec_in, test_target = test_data[0].to(de), test_data[1].to(de), test_data[2].to(de)
        test_score_f = get_scores(model_train, test_enc_in, test_dec_in, test_target,
                                  n_items_f, param, sas, domain, de)
        test_score_r = get_scores(model_train, test_enc_in, test_dec_in, test_target,
                                  n_items_r, param, sas, domain, de)
        rank_eval_f.extend(torch.argsort(torch.argsort(-eval_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_eval_r.extend(torch.argsort(torch.argsort(-eval_score_r, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_test_f.extend(torch.argsort(torch.argsort(-test_score_f, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_test_r.extend(torch.argsort(torch.argsort(-test_score_r, dim=1), dim=1)[:, 0].cpu().detach().numpy())
    # TODO metrics
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


def get_user_embed(model, seq, domain, param, device, pad_idx):
    mask = seq == pad_idx
    mask = (1 - mask.to(int)).view(-1).to(torch.float32).to(device)
    seq = seq.to(device)
    if torch.cuda.device_count() > 1:
        seq_embed = model.module.get_seq_embed(seq, domain=domain,
                                               mask=mask.view(-1, param.rec_maxlen))[:, -1, :]
    else:
        seq_embed = model.get_seq_embed(seq, domain=domain,
                                        mask=mask.view(-1, param.rec_maxlen))[:, -1, :]
    return seq_embed


def train_gan(netG, netD, rec_loader, opt_d, opt_g,
              device, param, iters, train_overlap, devices):
    l2_func = nn.DataParallel(l2_constraint(),
                              device_ids=devices)
    # k_val = [5, 10, 20, 30]
    # result_a = [{}, {}]
    # result_b = [{}, {}]
    # for val in k_val:
    #     result_a[0][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
    #                              "ht_test": [], "ndcg_test": [], "mrr_test": []}
    #     result_a[1][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
    #                              "ht_test": [], "ndcg_test": [], "mrr_test": []}
    #     result_b[0][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
    #                              "ht_test": [], "ndcg_test": [], "mrr_test": []}
    #     result_b[1][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
    #                              "ht_test": [], "ndcg_test": [], "mrr_test": []}
    # metrics_name = list(result_a[0]["5"].keys())

    FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
    a_iterator = iter(rec_loader[0])
    b_iterator = iter(rec_loader[1])
    overlap_iter = iter(train_overlap)
    one_label = torch.from_numpy(np.ones(param.batch_size, dtype=float)).cuda()
    zero_label = torch.from_numpy(np.zeros(param.batch_size, dtype=float)).cuda()
    loss_ce = nn.BCEWithLogitsLoss()
    for iteration in tqdm(range(iters)):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True   # they are set to False below in netG update

        for iter_d in range(CRITIC_ITERS):
            # a domain data
            try:
                in_seq_a, _, _, _, _, _, _, _ = get_next_batch(a_iterator, device)
            except StopIteration:
                # next epoch for a domain
                a_iterator = iter(rec_loader[0])
                in_seq_a, _, _, _, _, _, _, _ = get_next_batch(a_iterator, device)
            mask_a = in_seq_a == param.pad_index
            mask_a = (1 - mask_a.to(int)).view(-1).to(torch.float32).to(device)
            in_seq_a = in_seq_a.to(device)
            if torch.cuda.device_count() > 1:
                in_seq_ae = netG.module.get_seq_embed(in_seq_a, domain="a",
                                                      mask=mask_a.view(-1, param.rec_maxlen))[:, -1, :]
            else:
                in_seq_ae = netG.get_seq_embed(in_seq_a, domain="a",
                                               mask=mask_a.view(-1, param.rec_maxlen))[:, -1, :]
            in_seq_ae = in_seq_ae.detach()
            # b domain data
            try:
                in_seq_b, _, _, _, _, _, _, _ = get_next_batch(b_iterator, device)
            except StopIteration:
                # next epoch for b domain
                b_iterator = iter(rec_loader[1])
                in_seq_b, _, _, _, _, _, _, _ = get_next_batch(b_iterator, device)
            mask_b = in_seq_b == param.pad_index
            mask_b = (1 - mask_b.to(int)).view(-1).to(torch.float32).to(device)
            in_seq_b = in_seq_b.to(device)
            if torch.cuda.device_count() > 1:
                in_seq_be = netG.module.get_seq_embed(in_seq_b, domain="b",
                                                      mask=mask_b.view(-1, param.rec_maxlen))[:, -1, :]
            else:
                in_seq_be = netG.get_seq_embed(in_seq_b, domain="b",
                                               mask=mask_b.view(-1, param.rec_maxlen))[:, -1, :]
            in_seq_be = in_seq_be.detach()

            #  go through discriminator.
            opt_d.zero_grad()
            D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
            D_fake = netD(in_seq_be)
            # loss 1
            # real_loss = D_real.mean()
            # fake_loss = D_fake.mean()
            # dis_loss = fake_loss - real_loss
            # loss 2
            fake_loss, real_loss = loss_ce(D_fake, zero_label), loss_ce(D_real, one_label)
            dis_loss = fake_loss + real_loss
            dis_loss.backward()
            # train with gradient penalty  in_seq_ae.data
            gradient_penalty = calc_gradient_penalty(netD, in_seq_ae, in_seq_be,
                                                     param.batch_size, device)
            gradient_penalty.backward()
            # totoal_loss_d = dis_loss + gradient_penalty
            # totoal_loss_d.babackward()
            # D_cost = fake_loss - real_loss + gradient_penalty # loss_1

            D_cost = fake_loss + real_loss + gradient_penalty
            Wasserstein_D = D_real.mean() - D_fake.mean()
            opt_d.step()

        if not FIXED_GENERATOR:
            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            opt_g.zero_grad()
            # a domain data
            try:
                in_seq_a, dec_in_a, dec_out_a, n_items_a, _, _, bs, sl = get_next_batch(a_iterator, device)
            except StopIteration:
                a_iterator = iter(rec_loader[0])
                in_seq_a, dec_in_a, dec_out_a, n_items_a, _, _, bs, sl = get_next_batch(a_iterator, device)
            in_seq_ae = get_user_embed(netG, in_seq_a, "a", param, device, param.vocab_size_a)
            # b domain data
            try:
                in_seq_b, dec_in_b, dec_out_b, n_items_b, _, _, bs, sl = get_next_batch(b_iterator, device)
            except StopIteration:
                b_iterator = iter(rec_loader[1])
                in_seq_b, dec_in_b, dec_out_b, n_items_b, _, _, bs, sl = get_next_batch(b_iterator, device)
            in_seq_be = get_user_embed(netG, in_seq_b, "b", param, device, param.vocab_size_b)

            # =============== overlap data loader
            try:
                overlap_a, overlap_b = next(overlap_iter)
            except StopIteration:
                overlap_iter = iter(train_overlap)
                overlap_a, overlap_b = next(overlap_iter)
            over_seq_ae = get_user_embed(netG, overlap_a[0].to(device), "a", param, device, param.vocab_size_a)
            over_seq_be = get_user_embed(netG, overlap_b[0].to(device), "b", param, device, param.vocab_size_b)
            overlap_loss = l2_func(over_seq_ae, over_seq_be)
            # go through dis for gan loss
            D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
            D_fake = netD(in_seq_be)
            # loss 1
            # real_loss = D_real.mean()
            # fake_loss = D_fake.mean()
            # g_dis_loss = fake_loss - real_loss
            # loss 2
            fake_loss, real_loss = loss_ce(D_fake, one_label), loss_ce(D_real, zero_label)
            g_dis_loss = fake_loss + real_loss
            # g_dis_loss.backward()
            # ============ Reconstruction loss
            mask_a = get_pad_mask(dec_out_a, param.pad_index, device)
            mask_b = get_pad_mask(dec_out_b, param.pad_index, device)
            # ***** implement different losses for different tasks. *****
            loss_recon_a = loss_ae(netG, in_seq_a, dec_in_a, dec_out_a, n_items_a, True, bs, sl,
                                   param, mask_a, device, domain="a")
            loss_recon_b = loss_ae(netG, in_seq_b, dec_in_b, dec_out_b, n_items_b, True, bs, sl,
                                   param, mask_b, device, domain="b")
            # *****  recommendation loss: or just use only one of the recom and reconstruction loss *****
            # loss_recommend_a = loss_bpr_func(netG, in_seq_a, dec_in_a, dec_out_a, n_items_a, mask_a,
            #                                  "a", param)
            # loss_recommend_b = loss_bpr_func(netG, in_seq_b, dec_in_b, dec_out_b, n_items_b, mask_b,
            #                                  "b", param)

            # back propagation: 1:1:1
            g_dis_loss.backward()
            loss_recon_a.backward()
            loss_recon_b.backward()
            overlap_loss.backward()
            # loss_recommend_a.backward()
            # loss_recommend_b.backward()
            # Reconstruction loss:
            opt_g.step()

        # Write logs and save samples
        plot.plot(param.result_path + '/disc cost_%s' % date, D_cost.cpu().data.numpy())
        plot.plot(param.result_path + '/wasserstein distance_%s' % date, Wasserstein_D.cpu().data.numpy())
        plot.plot(param.result_path + '/join_recon_a_%s' % date, loss_recon_a.cpu().data.numpy())
        plot.plot(param.result_path + '/join_recon_b_%s' % date, loss_recon_b.cpu().data.numpy())

        if not FIXED_GENERATOR:
            plot.plot(param.result_path + '/gen cost_%s' % date, g_dis_loss.cpu().data.numpy())
        if iteration % 100 == 99:
            if not os.path.isdir(param.result_path + "/gan_loss"):
                os.mkdir(param.result_path + "/gan_loss")
            plot.flush(param.result_path + "/gan_loss")
        plot.tick()


def load_batch_data(data_iterator, data, device):
    try:
        enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(data_iterator, device)
    except StopIteration:
        data_iterator = iter(data)
        enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(data_iterator, device)
    return enc_in, dec_in, dec_out, n_items, data_iterator


def get_pad_mask(seq, pad_index, device):
    mask = seq == pad_index
    mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
    return mask.to(device)


def train_gan_all(netG, netD, gan_loader, opt_d, opt_g, device, param, iterations,
                  train_overlap, rec_loaders, test_loaders, domain="a", overlap=True):
    if torch.cuda.device_count() > 1:
        l2_func = nn.DataParallel(l2_constraint())
    else:
        l2_func = l2_constraint()
    opt_final_rec = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.98))

    k_val = [5, 10, 20]
    result = [{}, {}]
    for val in k_val:
        result[0][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
        result[1][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
    metrics_name = list(result[0]["5"].keys())
    FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
    a_iterator = iter(gan_loader[0])
    b_iterator = iter(gan_loader[1])

    overlap_iter = iter(train_overlap)
    dataloader_iterator = iter(rec_loaders[0])

    if domain == "a":
        rec_iterator = iter(gan_loader[0])
    else:
        rec_iterator = iter(gan_loader[1])
    for iteration in tqdm(range(int(iterations * 1.2))):
        # gan training of the NetD and GUR encoder *** phase 2 ***
        if iteration < int(iterations * 0.6):  # or iteration % 2 == 0:
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():   # reset requires_grad
                p.requires_grad = True    # they are set to False below in netG update

            for iter_d in range(CRITIC_ITERS):
                # a domain data
                try:
                    in_seq_a, _, _, _, _, _, _, _ = get_next_batch(a_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        a_iterator = iter(gan_loader[0])  # freq
                    else:
                        a_iterator = iter(gan_loader[0])  # random
                    in_seq_a, _, _, _, _, _, _, _ = get_next_batch(a_iterator, device)
                mask_a = in_seq_a == param.pad_index
                mask_a = (1 - mask_a.to(int)).view(-1).to(torch.float32).to(device)
                in_seq_a = in_seq_a.to(device)
                if torch.cuda.device_count() > 1:
                    in_seq_ae = netG.module.get_seq_embed(in_seq_a, domain="a",
                                                          mask=mask_a.view(-1, param.rec_maxlen))[:, -1, :]
                else:
                    in_seq_ae = netG.get_seq_embed(in_seq_a, domain="a",
                                                   mask=mask_a.view(-1, param.rec_maxlen))[:, -1, :]
                in_seq_ae = in_seq_ae.detach()  # detach when training,
                # b domain data
                try:
                    in_seq_b, _, _, _, _, _, _, _ = get_next_batch(b_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        b_iterator = iter(gan_loader[1])
                    else:
                        b_iterator = iter(gan_loader[1])
                    in_seq_b, _, _, _, _, _, _, _ = get_next_batch(b_iterator, device)
                mask_b = in_seq_b == param.pad_index
                mask_b = (1 - mask_b.to(int)).view(-1).to(torch.float32).to(device)
                in_seq_b = in_seq_b.to(device)
                if torch.cuda.device_count() > 1:
                    in_seq_be = netG.module.get_seq_embed(in_seq_b, domain="b",
                                                          mask=mask_b.view(-1, param.rec_maxlen))[:, -1, :]
                else:
                    in_seq_be = netG.get_seq_embed(in_seq_b, domain="b",
                                                   mask=mask_b.view(-1, param.rec_maxlen))[:, -1, :]
                in_seq_be = in_seq_be.detach()

                # go through discriminator.
                opt_d.zero_grad()
                D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
                D_fake = netD(in_seq_be)
                # loss 1
                real_loss = D_real.mean()
                fake_loss = D_fake.mean()
                dis_loss = fake_loss - real_loss
                # loss 2
                # fake_loss, real_loss = loss_ce(D_fake, zero_label), loss_ce(D_real, one_label)
                # dis_loss = fake_loss + real_loss

                dis_loss.backward()
                # train with gradient penalty  in_seq_ae.data
                gradient_penalty = calc_gradient_penalty(netD, in_seq_ae, in_seq_be, param.batch_size, device)
                gradient_penalty.backward()

                D_cost = fake_loss - real_loss + gradient_penalty  # loss_1
                # D_cost = fake_loss + real_loss + gradient_penalty
                Wasserstein_D = D_real.mean() - D_fake.mean()
                opt_d.step()

            if not FIXED_GENERATOR:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid update discriminator
                opt_g.zero_grad()

                # ===================================================== discriminator loss
                # a-domain data
                try:
                    in_seq_a, dec_in_a, dec_out_a, n_items_a, _, _, bs, sl = get_next_batch(a_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        a_iterator = iter(gan_loader[0])  # freq a-domain data
                    else:
                        a_iterator = iter(gan_loader[0])  # rand a-domain data
                    in_seq_a, dec_in_a, dec_out_a, n_items_a, _, _, bs, sl = get_next_batch(a_iterator, device)
                    #  bs, sl are the same in different domain.
                in_seq_ae = get_user_embed(netG, in_seq_a, "a", param, device, param.vocab_size_a)
                # b domain data
                try:
                    in_seq_b, dec_in_b, dec_out_b, n_items_b, _, _, bs, sl = get_next_batch(b_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        b_iterator = iter(gan_loader[1])  # freq b-domain
                    else:
                        b_iterator = iter(gan_loader[1])  # random b-domain
                    in_seq_b, dec_in_b, dec_out_b, n_items_b, _, _, bs, sl = get_next_batch(b_iterator, device)
                in_seq_be = get_user_embed(netG, in_seq_b, "b", param, device, param.vocab_size_b)

                D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
                D_fake = netD(in_seq_be)
                # loss 1
                real_loss = D_real.mean()
                fake_loss = D_fake.mean()
                g_dis_loss = real_loss - fake_loss
                # loss 2
                # fake_loss, real_loss = loss_ce(D_fake, one_label), loss_ce(D_real, zero_label)
                # g_dis_loss = fake_loss + real_loss
                g_dis_loss.backward()

                # ===================================================== l2 loss on overlap data loader
                if overlap:
                    try:
                        overlap_a, overlap_b = next(overlap_iter)
                    except StopIteration:
                        overlap_iter = iter(train_overlap)
                        overlap_a, overlap_b = next(overlap_iter)
                    over_seq_ae = get_user_embed(netG, overlap_a[0].to(device), "a", param, device, param.vocab_size_a)
                    over_seq_be = get_user_embed(netG, overlap_b[0].to(device), "b", param, device, param.vocab_size_b)
                    # overlap_loss = l2_loss(over_seq_ae, over_seq_be)
                    if torch.cuda.device_count() > 1:
                        overlap_loss = l2_func.module.forward_2(over_seq_ae, over_seq_be)
                    else:
                        overlap_loss = l2_func.forward_2(over_seq_ae, over_seq_be)
                    overlap_loss.backward()
                # ======== reconstruction loss
                mask_rec_a = get_pad_mask(dec_out_a, param.pad_index, device)
                loss_recon_a = loss_ae(netG, in_seq_a, dec_in_a, dec_out_a, n_items_a, True, bs, sl,
                                       param, mask_rec_a, device, domain="a")

                mask_rec_b = get_pad_mask(dec_out_b, param.pad_index, device)
                loss_recon_b = loss_ae(netG, in_seq_b, dec_in_b, dec_out_b, n_items_b, True, bs, sl,
                                       param, mask_rec_b, device, domain="b")

                loss_recon_a.backward()
                loss_recon_b.backward()

                # ===================================================== Recommendation loss on target domain
                # None.
                # back-propagation
                opt_g.step()
            plot.plot(param.result_path + '/disc cost_%s' % date, D_cost.cpu().data.numpy())
            plot.plot(param.result_path + '/wasserstein distance_%s' % date, Wasserstein_D.cpu().data.numpy())
            plot.plot(param.result_path + '/join_recon_a%s' % date, loss_recon_a.cpu().data.numpy())
            plot.plot(param.result_path + '/join_recon_b%s' % date, loss_recon_b.cpu().data.numpy())
            plot.plot(param.result_path + '/gen cost_%s' % date, g_dis_loss.cpu().data.numpy())
        else:  # tune with recommendation task in target domain. *** phase 3 ***
            opt_final_rec.zero_grad()
            try:
                enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)
            except StopIteration:
                if iteration > int(iterations * 0.8):  # freq
                    dataloader_iterator = iter(rec_loaders[1])
                else:  # random
                    dataloader_iterator = iter(rec_loaders[0])

                enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)
            try:
                in_seq_rec, dec_in_rec, dec_out_rec, n_items_rec, _, _, bs, sl = get_next_batch(rec_iterator,
                                                                                                device)
            except StopIteration:
                if domain == "a":
                    rec_iterator = iter(gan_loader[0])  # rand a-domain data
                else:
                    rec_iterator = iter(gan_loader[1])  # rand a-domain data
                in_seq_rec, dec_in_rec, dec_out_rec, n_items_rec, _, _, bs, sl = get_next_batch(rec_iterator,
                                                                                                device)
            # rec
            mask_rec = get_pad_mask(dec_out_rec, param.pad_index, device)
            loss_recon_rec = loss_ae(netG, in_seq_rec, dec_in_rec, dec_out_rec, n_items_rec, True, bs, sl,
                                     param, mask_rec, device, domain=domain)
            loss_recon_rec.backward()

            # mask
            mask = dec_out == param.pad_index
            mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
            mask.to(device)
            loss_recommend = loss_bpr_func(netG, enc_in, dec_in, dec_out, n_items, mask, domain, param)
            # back propagation
            loss_recommend.backward()
            # back propagation: 1:1:1:1 for discriminator, l2, reconstruction and recommendation loss
            opt_final_rec.step()
            # opt_g.step()

            plot.plot(param.result_path + '/tuning_recommendation_loss', loss_recommend.cpu().data.numpy())

        # Write logs and save samples
        if iteration > int(iterations * 0.8) and iteration % 30 == 29:
            netG.eval()
            result_tmp = evaluation_2(netG, test_loaders, device, param,
                                      sas=False, domain=domain)
            for key in k_val:
                key = str(key)
                for metric in metrics_name:
                    result[0][key][metric].extend(result_tmp[0][key][metric])
                    result[1][key][metric].extend(result_tmp[1][key][metric])
            Dataloader.save_pickle(result, param.result_path + "/result_%s.pickle" % param.target_domain)
            netG.train()
        # if not FIXED_GENERATOR:
        #     plot.plot(param.result_path + '/gen cost_%s' % date, g_dis_loss.cpu().data.numpy())
        if iteration % 100 == 99:
            if not os.path.isdir(param.result_path + "/gan_loss"):
                os.mkdir(param.result_path + "/gan_loss")
            plot.flush(param.result_path + "/gan_loss")
        plot.tick()


def train_gan_all_2(netG, netD, gan_loader, opt_d, opt_g, device, param, iters,
                    train_overlap, rec_loaders, test_loaders, domain="a", devices=None):
    l2_func = nn.DataParallel(l2_constraint(),
                              device_ids=devices)
    opt_final_rec = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.9, 0.98))

    k_val = [5, 10, 20, 30]
    result = [{}, {}]
    for val in k_val:
        result[0][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
        result[1][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
    metrics_name = list(result[0]["5"].keys())
    FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
    a_iterator = iter(gan_loader[0])
    b_iterator = iter(gan_loader[1])

    overlap_iter = iter(train_overlap)
    dataloader_iterator = iter(rec_loaders[0])

    one_label = torch.from_numpy(np.ones(param.batch_size, dtype=float)).cuda()
    zero_label = torch.from_numpy(np.zeros(param.batch_size, dtype=float)).cuda()
    loss_ce = nn.BCEWithLogitsLoss(reduction="mean")

    for iteration in tqdm(range(int(iters * 1.5))):
        # print("gan iteration,", iteration)
        ############################
        # (1) Update D network
        ###########################
        if iteration < int(iters * 0.8):  # or iteration % 2 == 0:
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(CRITIC_ITERS):
                # a domain data
                try:
                    in_seq_a, _, _, _, _, _, _, _ = get_next_batch(a_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        a_iterator = iter(gan_loader[0])  # freq
                    else:
                        a_iterator = iter(gan_loader[0])  # random
                    in_seq_a, _, _, _, _, _, _, _ = get_next_batch(a_iterator, device)
                mask_a = in_seq_a == param.pad_index
                mask_a = (1 - mask_a.to(int)).view(-1).to(torch.float32).to(device)
                in_seq_a = in_seq_a.to(device)
                if torch.cuda.device_count() > 1:
                    in_seq_ae = netG.module.get_seq_embed(in_seq_a, domain="a",
                                                          mask=mask_a.view(-1, param.rec_maxlen))[:, -1, :]
                else:
                    in_seq_ae = netG.get_seq_embed(in_seq_a, domain="a",
                                                   mask=mask_a.view(-1, param.rec_maxlen))[:, -1, :]
                # in_seq_ae = in_seq_ae.detach()  # detach when training,
                # b domain data
                try:
                    in_seq_b, _, _, _, _, _, _, _ = get_next_batch(b_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        b_iterator = iter(gan_loader[1])
                    else:
                        b_iterator = iter(gan_loader[1])
                    in_seq_b, _, _, _, _, _, _, _ = get_next_batch(b_iterator, device)
                mask_b = in_seq_b == param.pad_index
                mask_b = (1 - mask_b.to(int)).view(-1).to(torch.float32).to(device)
                in_seq_b = in_seq_b.to(device)
                if torch.cuda.device_count() > 1:
                    in_seq_be = netG.module.get_seq_embed(in_seq_b, domain="b",
                                                          mask=mask_b.view(-1, param.rec_maxlen))[:, -1, :]
                else:
                    in_seq_be = netG.get_seq_embed(in_seq_b, domain="b",
                                                   mask=mask_b.view(-1, param.rec_maxlen))[:, -1, :]
                in_seq_ae = in_seq_ae.detach()
                in_seq_be = in_seq_be.detach()

                # go through discriminator.
                opt_d.zero_grad()
                D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
                D_fake = netD(in_seq_be)
                # loss 1
                # real_loss = D_real.mean()
                # fake_loss = D_fake.mean()
                # dis_loss = fake_loss - real_loss
                # loss 2
                fake_loss, real_loss = loss_ce(D_fake, zero_label), loss_ce(D_real, one_label)
                dis_loss = fake_loss + real_loss

                dis_loss.backward()

                # train with gradient penalty  in_seq_ae.data
                gradient_penalty = calc_gradient_penalty(netD, in_seq_ae, in_seq_be, param.batch_size, device)
                gradient_penalty.backward()

                # D_cost = fake_loss - real_loss + gradient_penalty    # loss_1
                D_cost = fake_loss + real_loss + gradient_penalty
                Wasserstein_D = D_real.mean() - D_fake.mean()
                opt_d.step()

            if not FIXED_GENERATOR:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid update discriminator
                opt_g.zero_grad()

                # ===================================================== discriminator loss
                # a-domain data
                try:
                    in_seq_a, dec_in_a, dec_out_a, n_items_a, _, _, bs, sl = get_next_batch(a_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        a_iterator = iter(gan_loader[0])  # freq a-domain data
                    else:
                        a_iterator = iter(gan_loader[0])  # rand a-domain data
                    in_seq_a, dec_in_a, dec_out_a, n_items_a, _, _, bs, sl = get_next_batch(a_iterator, device)
                    # bs, sl are the same in different domain.
                in_seq_ae = get_user_embed(netG, in_seq_a, "a", param, device, param.vocab_size_a)
                # b domain data
                try:
                    in_seq_b, dec_in_b, dec_out_b, n_items_b, _, _, bs, sl = get_next_batch(b_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        b_iterator = iter(gan_loader[1])  # freq b-domain
                    else:
                        b_iterator = iter(gan_loader[1])  # random b-domain
                    in_seq_b, dec_in_b, dec_out_b, n_items_b, _, _, bs, sl = get_next_batch(b_iterator, device)
                in_seq_be = get_user_embed(netG, in_seq_b, "b", param, device, param.vocab_size_b)

                D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
                D_fake = netD(in_seq_be)
                # loss 1
                # real_loss = D_real.mean()
                # fake_loss = D_fake.mean()
                # g_dis_loss = real_loss - fake_loss
                # loss 2
                fake_loss, real_loss = loss_ce(D_fake, zero_label), loss_ce(D_real, one_label)
                g_dis_loss = - (fake_loss + real_loss)
                g_dis_loss.backward()

                # ===================================================== l2 loss on overlap data loader
                try:
                    overlap_a, overlap_b = next(overlap_iter)
                except StopIteration:
                    overlap_iter = iter(train_overlap)
                    overlap_a, overlap_b = next(overlap_iter)
                over_seq_ae = get_user_embed(netG, overlap_a[0].to(device), "a", param, device, param.vocab_size_a)
                over_seq_be = get_user_embed(netG, overlap_b[0].to(device), "b", param, device, param.vocab_size_b)
                # overlap_loss = l2_loss(over_seq_ae, over_seq_be)
                if torch.cuda.device_count() > 1:
                    overlap_loss = l2_func.module.forward_2(over_seq_ae, over_seq_be)
                else:
                    overlap_loss = l2_func.forward_2(over_seq_ae, over_seq_be)
                overlap_loss.backward()

                # ===================================================== Reconstruction loss on target domain
                # try:
                #     enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)
                # except StopIteration:
                #     if domain == "b" and iteration > 1000 or iteration > 1500:
                #         dataloader_iterator = iter(rec_loaders[1])
                #     else:
                #         dataloader_iterator = iter(rec_loaders[0])
                #     enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)

                mask_rec_a = get_pad_mask(dec_out_a, param.pad_index, device)
                loss_recon_a = loss_ae(netG, in_seq_a, dec_in_a, dec_out_a, n_items_a, True, bs, sl,
                                       param, mask_rec_a, device, domain="a")

                mask_rec_b = get_pad_mask(dec_out_b, param.pad_index, device)
                loss_recon_b = loss_ae(netG, in_seq_b, dec_in_b, dec_out_b, n_items_b, True, bs, sl,
                                       param, mask_rec_b, device, domain="b")

                loss_recon_a.backward()
                loss_recon_b.backward()

                opt_g.step()

            plot.plot(param.result_path + '/disc cost_%s' % date, D_cost.cpu().data.numpy())
            plot.plot(param.result_path + '/wasserstein distance_%s' % date, Wasserstein_D.cpu().data.numpy())
            plot.plot(param.result_path + '/join_recon_a%s' % date, loss_recon_a.cpu().data.numpy())
            plot.plot(param.result_path + '/join_recon_b%s' % date, loss_recon_b.cpu().data.numpy())
            plot.plot(param.result_path + '/gen cost_%s' % date, g_dis_loss.cpu().data.numpy())
        else:  # Recommendation
            opt_final_rec.zero_grad()
            try:
                enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)
            except StopIteration:
                if domain == "b" and iteration > int(iters * 0.8) + 300:  # freq
                    dataloader_iterator = iter(rec_loaders[1])
                else:  # random
                    if iteration > int(iters * 0.8) + 500:
                        dataloader_iterator = iter(rec_loaders[1])
                    else:
                        dataloader_iterator = iter(rec_loaders[0])

                enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)
            # mask
            mask = dec_out == param.pad_index
            mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
            mask.to(device)
            loss_recommend = loss_bpr_func(netG, enc_in, dec_in, dec_out, n_items, mask, domain, param)
            # back propagation
            loss_recommend.backward()
            # back propagation: 1:1:1:1 for discriminator, l2, reconstruction and recommendation loss
            opt_final_rec.step()
            plot.plot(param.result_path + '/tuning_recommendation_loss', loss_recommend.cpu().data.numpy())

        # Write logs and save samples
        if iteration > 200 and iteration % 60 == 59:
            netG.eval()
            result_tmp = evaluation_2(netG, test_loaders, device, param,
                                      sas=False, domain=domain)
            for key in k_val:
                key = str(key)
                for metric in metrics_name:
                    result[0][key][metric].extend(result_tmp[0][key][metric])
                    result[1][key][metric].extend(result_tmp[1][key][metric])
            Dataloader.save_pickle(result, param.result_path + "/result_%s.pickle" % param.target_domain)
            netG.train()

        if iteration % 100 == 99:
            if not os.path.isdir(param.result_path + "/gan_loss"):
                os.mkdir(param.result_path + "/gan_loss")
            plot.flush(param.result_path + "/gan_loss")
        plot.tick()


def train_recon_x(model_train, opt, steps, data, param, device, neg_sample=True,
                  loss_type="s_soft", opt_type="org"):
    """
    :type param: object
    :param model_train:
    :param opt:
    :param steps:
    :param data:
    :param param:
    :param device:
    :param neg_sample:
    :param loss_type:
    :param opt_type:
    :return:
    loss_type == "s_soft"时，训练auto encoder的重构loss训练。
    """
    model_train.train()
    data_iterator_a = iter(data[0])  # a domain data
    data_iterator_b = iter(data[1])  # b domain data
    seqs, _, _, _ = next(data_iterator_a)
    bs, sl = seqs[2].shape[0], seqs[2].shape[1]
    for i in tqdm(range(steps)):
        enc_in_a, dec_in_a, dec_out_a, n_items_a, data_iterator_a = load_batch_data(data_iterator_a, data[0], device)
        enc_in_b, dec_in_b, dec_out_b, n_items_b, data_iterator_b = load_batch_data(data_iterator_b, data[1], device)
        # masks
        mask_a = get_pad_mask(dec_out_a, param.pad_index, device)
        mask_b = get_pad_mask(dec_out_b, param.pad_index, device)
        opt.zero_grad()
        if loss_type == "s_soft":
            # selected softmax: reconstruction loss for autoencoder training.
            loss_a = loss_ae(model_train, enc_in_a, dec_in_a, dec_out_a, n_items_a,
                             neg_sample, bs, sl, param, mask_a, device, domain="a")
            loss_b = loss_ae(model_train, enc_in_b, dec_in_b, dec_out_b, n_items_b,
                             neg_sample, bs, sl, param, mask_b, device, domain="b")
        elif loss_type == "bpr":
            # BPR loss for training the recommendation loss
            loss_a = loss_bpr_func(model_train, enc_in_a, dec_in_a, dec_out_a, n_items_a, mask_a, param)
            loss_b = loss_bpr_func(model_train, enc_in_b, dec_in_b, dec_out_b, n_items_b, mask_b, param)
        else:
            print("loss configuration error")
            sys.exit()
        # back propagation
        loss_a.backward()
        loss_b.backward()
        # update
        if opt_type == "org":  # normal optimizer (Adam)
            opt.step()
        else:  # scheduled optimizer
            opt.step_and_update_lr()
        # Get loss:
        loss_epoch_a = loss_a.item()
        loss_epoch_b = loss_b.item()
        # eval step.
        if i % 50 == 49:
            if loss_type == "s_soft":
                print("reconstruction loss after %d batch" % i, loss_epoch_a, loss_epoch_b)
                plot.plot(param.result_path + '/reconstruct_loss_a_%s' %
                          param.date,
                          loss_epoch_a)
                plot.plot(param.result_path + '/reconstruct_loss_b_%s' %
                          param.date,
                          loss_epoch_b)
            else:
                print("BPR loss after %d batch" % i, loss_epoch_a, loss_epoch_b)
                plot.plot(param.result_path + '/bpr_loss_a_%s' %
                          param.date,
                          loss_epoch_a)
                plot.plot(param.result_path + '/bpr_loss_b_%s' %
                          param.date,
                          loss_epoch_b)

            # leave the evaluation to another function: implement
            # TODO: implement early-stop here.
            plot.flush(param.result_path)
            plot.tick()


def recommendation_tune(model, rec_loader, test_loader, steps, param, device, domain):
    """
    :param domain:
    :param device:
    :param model:        pre-trained model
    :param rec_loader:   recommendation data loader of a certain domain; [rand_rec_loader, freq_rec_loader]
    :param test_loader:  recommendation data loader of a certain domain; [rand_rec_loader, freq_rec_loader]
    :param steps:        number of fine-tuning steps.
    :param param:        all kinds of hyper parameters.
    :return:             None, save corresponding test result.
    """

    k_val = [5, 10, 20, 30]
    model.train()

    dataloader_iterator = iter(rec_loader[0])

    result = [{}, {}]
    for val in k_val:
        result[0][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
        result[1][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
    k_val_str = list(result[0].keys())
    metrics_name = list(result[0][k_val_str[0]].keys())
    opt = optim.Adam(model.parameters(), lr=0.006, betas=(0.9, 0.9))

    for i in tqdm(range(steps)):
        try:
            enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)
        except StopIteration:
            if domain == "b" and i > int(steps / 2):
                dataloader_iterator = iter(rec_loader[1])
            else:
                dataloader_iterator = iter(rec_loader[0])
            enc_in, dec_in, dec_out, n_items, _, _, bs, sl = get_next_batch(dataloader_iterator, device)
        # mask
        mask = dec_in == param.pad_index
        mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
        mask.to(device)
        opt.zero_grad()
        loss = loss_bpr_func(model, enc_in, dec_in, dec_out, n_items, mask, domain, param)
        # back propagation
        loss.backward()
        # update
        opt.step()  # normal optimizer (Adam)
        torch.cuda.empty_cache()
        if device == "cpu":
            loss_epoch = loss.detach().numpy()
        else:
            loss_epoch = loss.cpu().detach().numpy()
        # result_tmp = evaluation(model_train, eval_loader, device, param, sas=sas)
        # eval step.
        if i % param.eval_step == (param.eval_step - 1):
            if i > param.eval_step * 10:
                param.eval_step *= 2
            model.eval()
            print("BPR loss after %d batch" % i, loss_epoch)
            plot.plot(param.result_path + '/bpr_loss_%s' %
                      domain,
                      loss_epoch)
            # evaluate
            result_tmp = evaluation_2(model, test_loader, device, param, sas=False, domain=domain)
            print("eval HT@10 %f, test HT@10 %f" % (result_tmp[0]["10"]["ht_eval"][0],
                                                    result_tmp[0]["10"]["ht_test"][0]))
            print("eval HT@10 %f, test HT@10 %f" % (result_tmp[1]["10"]["ht_eval"][0],
                                                    result_tmp[1]["10"]["ht_test"][0]))
            for key in k_val_str:
                for metric in metrics_name:
                    result[0][key][metric].extend(result_tmp[0][key][metric])
                    result[1][key][metric].extend(result_tmp[1][key][metric])
            Dataloader.save_pickle(result, param.result_path + "/result_%s.pickle" % domain)
            model.train()
            plot.flush(param.result_path)
            plot.tick()


# def main(auto_cross, opt_rec, netD, opt_gen, opt_dis, param, device_t,
#          ae_loaders, rec_loaders, test_loaders, train_overlap):
#     # param, device_t, train_loader_ae, train_loader_re, eval_loader, item_freq = None, shared = True
#     print("============ Reconstruction pre-training.")
#     train_recon_x(auto_cross, opt_rec, param.n_warmup_steps, ae_loaders, param, device_t,
#                   neg_sample=True, loss_type="s_soft", opt_type="schedule")
#     # train_gan(auto_cross, netD, rec_loaders, test_loaders, opt_dis, opt_gen, device_t, param, 5000)
#     print("============ Adversarial training.")
#     train_gan(auto_cross, netD, ae_loaders, opt_dis, opt_gen, device_t, param, param.n_warmup_steps, train_overlap)
#
#     # save checkpoint here,
#     torch.save(auto_cross.state_dict(), param.model_path + "/pre_model")
#     # fine-tune on recommendation task. a domain and b domain separately.
#     # tune in domain "a"
#     print("============ Fine-tune in domain a.")
#     recommendation_tune(auto_cross, [rec_loaders[0][0], rec_loaders[1][0]],
#                         test_loaders[0], param.training_steps_tune,
#                         param, device_t, domain="a")
#     # tune in domain "b
#     print("============ Fine-tune in domain b.")
#     auto_cross.load_state_dict(torch.load(param.model_path + "/pre_model"))
#     recommendation_tune(auto_cross, [rec_loaders[0][1], rec_loaders[1][1]], test_loaders[1],
#                         param.training_steps_tune,
#                         param, device_t, domain="b")


def main_2(auto_cross, opt_rec, netD, opt_gen, opt_dis, param, device_t,
           ae_loaders, rec_loaders, test_loaders, train_overlap):
    print("============ Reconstruction pre-training (Phase 1).")
    train_recon_x(auto_cross, opt_rec, 200, ae_loaders, param, device_t,
                  neg_sample=True, loss_type="s_soft", opt_type="schedule")  # 5k
    if torch.cuda.device_count() > 1:
        torch.save(auto_cross.module.state_dict(), param.model_path + "/pre_model")
    else:
        torch.save(auto_cross.state_dict(), param.model_path + "/pre_model")
    print("============ Adversarial and recommendation training (phase 2 and phase 3).")
    train_gan_all(auto_cross, netD, ae_loaders, opt_dis, opt_gen,
                  device_t, param, param.training_steps_tune, train_overlap,
                  rec_loaders, test_loaders, domain=param.target_domain, overlap=False)


# def main_3(auto_cross, opt_rec, netD, opt_gen, opt_dis, param, device_t,
#            ae_loaders, rec_loaders, test_loaders, train_overlap, devices):
#     print("============ Reconstruction pre-training.")
#     # if os.path.isfile(param.model_path + "/pre_model"):
#     #     print("restore pre-trained model")
#     #     auto_cross.module.load_state_dict(torch.load(param.model_path + "/pre_model"))
#     # else:
#     train_recon_x(auto_cross, opt_rec, 2000, ae_loaders, param, device_t,
#                   neg_sample=True, loss_type="s_soft", opt_type="schedule")
#     torch.save(auto_cross.module.state_dict(), param.model_path + "/pre_model")
#     print("============ Adversarial and recommendation training.")
#     train_gan_all_2(auto_cross, netD, ae_loaders, opt_dis, opt_gen,
#                     device_t, param, param.training_steps_tune, train_overlap,
#                     rec_loaders, test_loaders, domain=param.target_domain,
#                     devices=devices)
