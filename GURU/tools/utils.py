# -*- coding: UTF-8 -*-
import matplotlib
import torch
import torch.nn as nn
from tqdm import tqdm
from .lossfunctions import SampledCrossEntropyLoss, BPRLoss
from .metrics import hit_at_k_batch, NDCG_at_k_batch, mrr_at_k_batch

matplotlib.use('Agg')

loss_s_ce = SampledCrossEntropyLoss(reduction="none")
loss_bpr = BPRLoss()


def get_next_batch(dataloader_iterator, device):
    """
    For training auttoencoder.
    :param device:
    :param dataloader_iterator:
    :return:
    """
    seqs, n_items, val, test = next(dataloader_iterator)
    n_items, val, test = n_items.to(device), val.to(device), test.to(device)
    enc_in, dec_in, dec_out = seqs[0], seqs[1], seqs[2]
    bs, sl = dec_out.shape[0], dec_out.shape[1]
    enc_in, dec_in, dec_out = enc_in.to(device), dec_in.to(device), dec_out.to(device)
    return enc_in, dec_in, dec_out, n_items, val, test, bs, sl


class Discriminator(nn.Module):

    def __init__(self, in_dim, out_dim, mid_dim):
        """
        :param in_dim:
        :param out_dim:
        :param mid_dim:
        TODO: specify number of hidden layers.
        """
        super(Discriminator, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        main = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(mid_dim, mid_dim * 2),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(mid_dim * 2, mid_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(mid_dim, out_dim),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


def loss_ae(model, enc_in, dec_in, dec_out, n_items, neg_sample, bs, sl, param, mask, device, domain="a"):
    """
    :param device:
    :param domain:
    :param mask:
    :param model:  Autoencoder model
    :param enc_in:
    :param dec_in:
    :param dec_out:
    :param n_items:
    :param neg_sample:
    :param bs:
    :param sl:
    :param param:
    :return:
    """
    logits = model(enc_in, dec_in, dec_out, n_items, domain, mask)
    if neg_sample:
        # negative sample use fake label
        label = torch.zeros(bs * sl).long().to(device)
        num_class = param.n_negs + 1
    else:
        label = dec_out.view(-1, 1).long()
        num_class = param.vocab_size
    loss_m = loss_s_ce(logits, label, num_class, mask=mask)
    # loss = loss_ce(logits.view(-1, num_class), label)
    # loss_m = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
    return loss_m


def loss_bpr_func(model_train, enc_in, dec_in, dec_out, n_items, mask, domain, param):
    """
    :param param:
    :param domain:
    :param mask:         input padding mask
    :param model_train:  wrapper model.
    :param enc_in:
    :param dec_in:
    :param dec_out:
    :param n_items:
    :return:
    """
    if torch.cuda.device_count() > 1:
        recom_embeddings = model_train.module.recommend_forward(enc_in, dec_in, domain, mask.view(-1, param.rec_maxlen))
        recom_embeddings = recom_embeddings.view(-1, param.enc_maxlen, 1, param.d_model)
        if domain == "a":
            n_embeddings = model_train.module.src_emb_a(n_items).view(-1, param.rec_maxlen, param.n_bpr_neg,
                                                                      param.d_model)
            p_embeddings = model_train.module.src_emb_a(dec_out).view(-1, param.rec_maxlen, 1, param.d_model)
        else:
            n_embeddings = model_train.module.src_emb_b(n_items).view(-1, param.rec_maxlen, param.n_bpr_neg,
                                                                      param.d_model)
            p_embeddings = model_train.module.src_emb_b(dec_out).view(-1, param.rec_maxlen, 1, param.d_model)
    else:
        recom_embeddings = model_train.recommend_forward(enc_in, dec_in, domain, mask.view(-1, param.rec_maxlen))
        recom_embeddings = recom_embeddings.view(-1, param.enc_maxlen, 1, param.d_model)
        if domain == "a":
            n_embeddings = model_train.src_emb_a(n_items).view(-1, param.rec_maxlen, param.n_bpr_neg, param.d_model)
            p_embeddings = model_train.src_emb_a(dec_out).view(-1, param.rec_maxlen, 1, param.d_model)
        else:
            n_embeddings = model_train.src_emb_b(n_items).view(-1, param.rec_maxlen, param.n_bpr_neg, param.d_model)
            p_embeddings = model_train.src_emb_b(dec_out).view(-1, param.rec_maxlen, 1, param.d_model)

    p_logits = torch.squeeze(torch.matmul(recom_embeddings, p_embeddings.transpose(2, 3)), 2)  # [b, len, 1, 1]
    n_logits = torch.squeeze(torch.matmul(recom_embeddings, n_embeddings.transpose(2, 3)), 2)  # [b, len, 1, neg]

    loss = loss_bpr(p_logits, n_logits, mask=mask)
    return loss


def get_scores(model, enc_in, dec_in, target, n_items, param):
    dec_embedding = model.get_embedding(enc_in, dec_in)
    dec_embedding = dec_embedding[:, -1, :]  # [B, d_model]  pred_embedding
    n_embeddings = model.AutoEnc.src_emb(n_items).view(-1,
                                                       param.candidate_size,
                                                       param.d_model)
    p_embeddings = model.AutoEnc.src_emb(target).view(-1,
                                                      param.d_model)
    candidate_embeddings = torch.cat([torch.unsqueeze(p_embeddings, 1), n_embeddings], dim=1)  # [B, cand+ 1, d_model]
    scores = torch.matmul(torch.unsqueeze(dec_embedding, 1), candidate_embeddings.transpose(1, 2))
    # [B, 1, candidate_size + 1]
    return torch.squeeze(scores)


def evaluation(model_train, data_loader, de, param, k_val=None):
    if k_val is None:
        k_val = [1, 5, 10, 20, 30]
    model_train.eval()
    print("evaluation and test")
    rank_eval = []
    rank_test = []
    result_tmp = {}
    for val in k_val:
        result_tmp[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                                "ht_test": [], "ndcg_test": [], "mrr_test": []}
    eval_iterator = iter(data_loader)
    for _ in tqdm(range(param.eval_steps)):
        try:
            eval_data, test_data, n_items = next(eval_iterator)
        except StopIteration:
            eval_iterator = iter(data_loader)
            eval_data, test_data, n_items = next(eval_iterator)
        n_items = n_items.to(de)
        eval_enc_in, eval_dec_in, eval_target = eval_data[0].to(de), eval_data[1].to(de), eval_data[2].to(de)
        eval_score = get_scores(model_train, eval_enc_in, eval_dec_in, eval_target, n_items, param)
        # test
        test_enc_in, test_dec_in, test_target = test_data[0].to(de), test_data[1].to(de), test_data[2].to(de)
        test_score = get_scores(model_train, test_enc_in, test_dec_in, test_target, n_items, param)
        rank_eval.extend(torch.argsort(torch.argsort(-eval_score, dim=1), dim=1)[:, 0].cpu().detach().numpy())
        rank_test.extend(torch.argsort(torch.argsort(-test_score, dim=1), dim=1)[:, 0].cpu().detach().numpy())

    for k in k_val:
        result_tmp[str(k)]["ht_eval"].append(hit_at_k_batch(rank_eval, k))
        result_tmp[str(k)]["ht_test"].append(hit_at_k_batch(rank_test, k))
        result_tmp[str(k)]["ndcg_eval"].append(NDCG_at_k_batch(rank_eval, k))
        result_tmp[str(k)]["ndcg_test"].append(NDCG_at_k_batch(rank_test, k))
        result_tmp[str(k)]["mrr_eval"].append(mrr_at_k_batch(rank_eval, k))
        result_tmp[str(k)]["mrr_test"].append(mrr_at_k_batch(rank_test, k))
    return result_tmp
