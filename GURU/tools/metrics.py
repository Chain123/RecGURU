# -*- coding: UTF-8 -*-
import numpy as np


def hit_at_k(r, k):
    """
    Args:
        r: rank of relevant item.
        k: Number of result to consider.
    Returns:
        hit at k
    """
    if r < k:
        return 1
    else:
        return 0


def hit_at_k_batch_old(r, k):
    hit = 0
    for rank in r:
        hit += hit_at_k(rank, k)
    return float(hit) / len(r)


def hit_at_k_batch(r, k):
    """
    :param r: list of rank, [Batch, 1] or [Batch]
    :param k: int, top K
    :return: Hit number for this batch.
    """
    r = (np.array(r) < k) + 0
    return float(sum(r)) / len(r)


def NDCG_at_k(r, k):
    """
    Args:
        r: rank of relevant item.
        k: Number of result to consider.
    Returns:
        NDCG at k
    """
    if r < k:
        return 1 / np.log2(r + 2)
    else:
        return 0


def NDCG_at_k_batch(r, k):
    ndcg = 0
    for rank in r:
        ndcg += NDCG_at_k(rank, k)
    return float(ndcg) / len(r)


def mrr_at_k(r, k):
    if r < k:
        return 1.0 / (r + 1)
    else:
        return 0.0


def mrr_at_k_batch(r, k):
    mrr = 0
    for rank in r:
        mrr += mrr_at_k(rank, k)
    return float(mrr) / len(r)


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item_i, item_j in test_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()  # not useful when testing

        prediction_i, prediction_j = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(
            item_i, indices).cpu().numpy().tolist()

        gt_item = item_i[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)


def metrics_m(model, test_loader, k_val=None):
    if k_val is None:
        k_val = [1, 5, 10, 20, 30]
    rank_eval_f = []
    rank_eval_r = []
    rank_test_f = []
    rank_test_r = []
    result_rand = {}
    result_freq = {}
    for val in k_val:
        result_rand[str(val)] = {"ht": [], "ndcg": [], "mrr": []}
        result_freq[str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    #         result_freq[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
    #                                  "ht_test": [], "ndcg_test": [], "mrr_test": []}
    # TODO add item_j_f, item_j_r, freq, rand.
    # built another dataloader for this.
    # ===== eval or test
    for user, item_i, item_j in test_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()  # not useful when testing

        prediction_r, prediction_f = model(user, item_i, item_j)
        # print(prediction_r.shape)
        # print(prediction_f.shape)
        rank_r = torch.argsort(torch.argsort(-prediction_r, dim=1), dim=1)[:, 0].cpu().detach().numpy()
        rank_f = torch.argsort(torch.argsort(-prediction_f, dim=1), dim=1)[:, 0].cpu().detach().numpy()
        rank_eval_r.extend(rank_r)
        rank_eval_f.extend(rank_f)

    for k in k_val:
        result_rand[str(k)]["ht"].append(hit_at_k_batch(rank_eval_r, k))
        result_rand[str(k)]["ndcg"].append(NDCG_at_k_batch(rank_eval_r, k))
        result_rand[str(k)]["mrr"].append(mrr_at_k_batch(rank_eval_r, k))

        result_freq[str(k)]["ht"].append(hit_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["ndcg"].append(NDCG_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["mrr"].append(mrr_at_k_batch(rank_eval_f, k))

    return result_rand, result_freq


def metrics_em(merge_func, map_func, a_model, b_model, test_loader, k_val=None):
    if k_val is None:
        k_val = [1, 5, 10, 20, 30]
    result_rand = {}
    result_freq = {}
    for val in k_val:
        result_rand[str(val)] = {"ht": [], "ndcg": [], "mrr": []}
        result_freq[str(val)] = {"ht": [], "ndcg": [], "mrr": []}
    #         result_freq[str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
    #                                  "ht_test": [], "ndcg_test": [], "mrr_test": []}
    # TODO add item_j_f, item_j_r, freq, rand.
    # built another dataloader for this.
    rank_eval_r_over, rank_eval_f_over = eval_em_all(merge_func, map_func, a_model, b_model,
                                                     test_loader[0], overlap=True)
    rank_eval_r_only, rank_eval_f_only = eval_em_all(merge_func, map_func, a_model, b_model,
                                                     test_loader[1], overlap=False)
    rank_eval_f = rank_eval_f_over + rank_eval_f_only
    rank_eval_r = rank_eval_r_over + rank_eval_r_only
    for k in k_val:
        result_rand[str(k)]["ht"].append(hit_at_k_batch(rank_eval_r, k))
        result_rand[str(k)]["ndcg"].append(NDCG_at_k_batch(rank_eval_r, k))
        result_rand[str(k)]["mrr"].append(mrr_at_k_batch(rank_eval_r, k))

        result_freq[str(k)]["ht"].append(hit_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["ndcg"].append(NDCG_at_k_batch(rank_eval_f, k))
        result_freq[str(k)]["mrr"].append(mrr_at_k_batch(rank_eval_f, k))

    return result_rand, result_freq


def eval_em_all(merge_func, map_func, a_model, b_model, test_loader, overlap=True):
    rank_eval_f = []
    rank_eval_r = []
    for user, item_i, item_j in test_loader:  # one overlap
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()  # not useful when testing

        # prediction_r, prediction_f = model(user, item_i, item_j)  # TODO replace this
        if overlap:
            user = merge_func(a_model.embed_user(user), b_model.embed_user(user))
        else:
            user = merge_func(a_model.embed_user(user), map_func(a_model.embed_user(user)))

        item_i = a_model.embed_item(item_i)
        item_j = a_model.embed_item(item_j)
        prediction_i = torch.squeeze(torch.matmul(torch.unsqueeze(user, 1), item_i.transpose(1, 2)))
        prediction_j = torch.squeeze(torch.matmul(torch.unsqueeze(user, 1), item_j.transpose(1, 2)))

        rank_r = torch.argsort(torch.argsort(-prediction_i, dim=1), dim=1)[:, 0].cpu().detach().numpy()
        rank_f = torch.argsort(torch.argsort(-prediction_j, dim=1), dim=1)[:, 0].cpu().detach().numpy()
        rank_eval_r.extend(rank_r)
        rank_eval_f.extend(rank_f)
    return rank_eval_r, rank_eval_f
