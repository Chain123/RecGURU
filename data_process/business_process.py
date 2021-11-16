# -*- coding: UTF-8 -*-

"""
business data process.
    1. unit item ids in both domains.
    2. 去掉连续出现的重复元素，在此基础上，再次过滤用户行为数据少于5的。
    3. form a cross-domain scenario with a given overlapping rate.
"""

import os
import gzip
import _pickle as pickle
import sys
from itertools import groupby
import argparse
import time
import numpy as np


def load_pickle(filename):
    f = open(filename, "rb")
    return pickle.load(f)  # encoding="latin"


def id_dict(data_path):
    a_num = 1
    b_num = 1
    a_dict = {}
    b_dict = {}
    files = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
    # read 
    for file_t in files:
        print("processing", file_t)
        f = gzip.open(file_t, 'rb')
        for line in f:
            data = line.decode("utf-8").strip().split("|")
            wesee_items = data[1].split(",")
            video_items = data[4].split(",")
            for item in wesee_items:
                if item not in a_dict:
                    a_dict[item] = a_num
                    a_num += 1

            for item in video_items:
                if item not in b_dict:
                    b_dict[item] = b_num
                    b_num += 1
    save_pickle(a_dict, "a_domain_ids.pickle")
    save_pickle(b_dict, "b_domain_ids.pickle")
    print("a domain number of items", a_num)
    print("b domain number of items", b_num)


def save_pickle(dict_name, file_name):
    with open(file_name, "wb") as fid:
        pickle.dump(dict_name, fid, -1)


def unique(in_list):
    res = [val[0] for val in groupby(in_list)]
    return res


def remove_repeats(data_path, output_path):
    # TODO get item frequency here.
    uid = 1
    a_valid = []
    b_valid = []

    a_num = 1
    b_num = 1
    a_dict = {}
    b_dict = {}

    item_freq_a = {}
    item_freq_b = {}
    for file_num in range(19):
        if file_num < 10:
            files = os.path.join(data_path, "00000%d_0.gz" % file_num)
        else:
            files = os.path.join(data_path, "0000%d_0.gz" % file_num)
        outfile = os.path.join(output_path, "processed_%d.pickle" % file_num)
        result_tmp = {"uid": [], "a_item": [], "b_item": []}
        print("processing", files)
        f = gzip.open(files, 'rb')

        for line in f:
            data = line.decode("utf-8").strip().split("|")
            wesee_items = data[1].split(",")
            video_items = data[4].split(",")
            wesee_items = unique(wesee_items)
            video_items = unique(video_items)
            result_tmp["uid"].append(uid)
            # result_tmp["a_item"].append(wesee_items)
            # result_tmp["b_item"].append(video_items)
            uid += 1
            if len(wesee_items) > 5:
                a_valid.append(uid)
                result_tmp["a_item"].append(wesee_items)
                for item in wesee_items:
                    if item not in a_dict:  # total items reid from 1 to N
                        a_dict[item] = a_num
                        a_num += 1
                    if item not in item_freq_a:
                        item_freq_a[item] = 0
                    item_freq_a[item] += 1
            else:
                result_tmp["a_item"].append([])
            if len(video_items) > 5:
                result_tmp["b_item"].append(video_items)
                b_valid.append(uid)
                for item in video_items:
                    if item not in b_dict:
                        b_dict[item] = b_num
                        b_num += 1
                    if item not in item_freq_b:
                        item_freq_b[item] = 0
                    item_freq_b[item] += 1
            else:
                result_tmp["b_item"].append([])
        save_pickle(result_tmp, outfile)
        # number_file += 1
        print("a domain number of items", a_num)
        print("b domain number of items", b_num)

    save_pickle(item_freq_a, os.path.join(output_path, "item_freq_a.pickle"))
    save_pickle(item_freq_b, os.path.join(output_path, "item_freq_b.pickle"))

    save_pickle(a_dict, os.path.join(output_path, "a_domain_ids.pickle"))
    save_pickle(b_dict, os.path.join(output_path, "b_domain_ids.pickle"))

    save_pickle(a_valid, os.path.join(output_path, "a_valid.pickle"))
    save_pickle(b_valid, os.path.join(output_path, "b_valid.pickle"))
    print("valid number of user in a domain,", len(a_valid))
    print("valid number of user in b domain,", len(b_valid))


"""
# Different overlap ratio and re-assign the item ids.
# a (wesee) 是target domain， 重合度 按照他的用户数量计算。
# a_total =
"""


def all_overlap_users(a_valid, b_valid):
    intersection_set = set.intersection(set(a_valid), set(b_valid))

    return list(intersection_set)


def str2id(list_t, dict_t):
    return [dict_t[val] for val in list_t]


def generate_data(data_path, ratio, output_path, num):
    # id_dict_a = load_pickle(os.path.join(file_path, "a_domain_ids.pickle"))
    # id_dict_b = load_pickle(os.path.join(file_path, "b_domain_ids.pickle"))
    # a_valid_user = load_pickle(os.path.join(file_path, "a_valid.pickle"))
    # b_valid_user = load_pickle(os.path.join(file_path, "b_valid.pickle"))
    info_path = "/data/ceph/seqrec/data/business"
    id_dict_a = load_pickle(os.path.join(data_path, "a_domain_ids.pickle"))
    id_dict_b = load_pickle(os.path.join(data_path, "b_domain_ids.pickle"))
    # a_valid_user = load_pickle(os.path.join(info_path, "a_all.pickle"))  # all users
    # b_valid_user = load_pickle(os.path.join(info_path, "b_all.pickle"))
    a_valid_user = load_pickle(os.path.join(file_path, "a_valid.pickle"))  # users with more than 5 inters.
    b_valid_user = load_pickle(os.path.join(file_path, "b_valid.pickle"))

    a_total = len(a_valid_user)
    b_total = len(b_valid_user)
    overlap_num = int(a_total * ratio)
    overlap_ids = all_overlap_users(a_valid_user, b_valid_user)
    print("a total", a_total)
    print("b total", b_total)
    print("number of all overlapped users", len(overlap_ids))

    # select "overlap_num" number of users from "overlap_ids" to satisfy the overlapping rate.
    if not os.path.isfile(os.path.join(info_path, "%d_over_userid.pickle" % int(overlap_ratio * 100))):
        np.random.shuffle(overlap_ids)
        if overlap_num < len(overlap_ids):
            select_overlap = overlap_ids[0:overlap_num]
        else:
            print("all overlapped are chosen")
            select_overlap = overlap_ids
        # save to file.
        save_pickle(select_overlap, os.path.join(info_path, "%d_over_userid.pickle" % int(overlap_ratio * 100)))
    else:  # loading exist file
        select_overlap = load_pickle(os.path.join(info_path, "%d_over_userid.pickle" % int(ratio * 100)))
    print("number of selected overlap users", len(select_overlap))
    a_only_data = {"seq": [], "val": [], "test": []}
    b_only_data = {"seq": [], "val": [], "test": []}
    overlap_data = {"seq_a": [], "val_a": [], "test_a": [],
                    "seq_b": [], "val_b": [], "test_b": []}
    files = [os.path.join(data_path, "processed_%d.pickle" % num)]
    file_num = 0
    wrong_overlap = 0
    for file_t in files:
        print(file_t)
        data_t = load_pickle(file_t)
        start = time.time()
        for u_id in range(len(data_t["uid"])):
            uid_t = data_t["uid"][u_id]
            wesee_items = str2id(data_t["a_item"][u_id], id_dict_a)
            video_items = str2id(data_t["b_item"][u_id], id_dict_b)
            if (u_id + 1) % 10000 == 9999:
                print("1w sample passed")
                print("time passed", time.time() - start)
                start = time.time()
            try:
                if uid_t in select_overlap:
                    if len(wesee_items) > 4 and len(video_items) > 4:
                        # print("index of user in selected overlapped ", select_overlap.index(uid_t))
                        # ID changing
                        overlap_data["seq_a"].append(wesee_items[:-2])
                        overlap_data["val_a"].append(wesee_items[-2])
                        overlap_data["test_a"].append(wesee_items[-1])

                        overlap_data["seq_b"].append(video_items[:-2])
                        overlap_data["val_b"].append(video_items[-2])
                        overlap_data["test_b"].append(video_items[-1])
                    else:
                        print("wrong overlap")
                        wrong_overlap += 1
                elif uid_t in overlap_ids:
                    # random 
                    if u_id % 2 == 0:
                        a_only_data["seq"].append(wesee_items[:-2])
                        a_only_data["val"].append(wesee_items[-2])
                        a_only_data["test"].append(wesee_items[-1])
                    else:
                        b_only_data["seq"].append(video_items[:-2])
                        b_only_data["val"].append(video_items[-2])
                        b_only_data["test"].append(video_items[-1])
                elif uid_t in a_valid_user and len(wesee_items) > 4:
                    a_only_data["seq"].append(wesee_items[:-2])
                    a_only_data["val"].append(wesee_items[-2])
                    a_only_data["test"].append(wesee_items[-1])

                elif uid_t in b_valid_user and len(video_items) > 4:
                    b_only_data["seq"].append(video_items[:-2])
                    b_only_data["val"].append(video_items[-2])
                    b_only_data["test"].append(video_items[-1])
            except Exception as e:
                print(data_t["uid"][u_id])
                print(data_t["a_item"][u_id])
                print(wesee_items)
                print(data_t["b_item"][u_id])
                print(video_items)
                sys.exit()
        file_num += 1
        # split each file into three parts. a_only, b_only, overlap.
        save_pickle(a_only_data, os.path.join(output_path, "a_only_%d.pickle" % num))
        save_pickle(b_only_data, os.path.join(output_path, "b_only_%d.pickle" % num))
        save_pickle(overlap_data, os.path.join(output_path, "overlap_%d.pickle" % num))
    print("a total", a_total)
    print("b total", b_total)
    print("number of all overlapped users", len(overlap_ids))
    print("wrong number of all overlapped users", wrong_overlap)


def statistic(data_path):
    # id_dict_a = load_pickle(os.path.join(data_path, "a_domain_ids.pickle"))
    # id_dict_b = load_pickle(os.path.join(data_path, "b_domain_ids.pickle"))
    # a_valid_user = load_pickle(os.path.join(data_path, "a_valid.pickle"))
    # b_valid_user = load_pickle(os.path.join(data_path, "b_valid.pickle"))
    files = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if "processed" in filename]
    # missed = 0
    # wrong = 0
    overlap = []
    a_all = []
    b_all = []
    for file_t in files:
        print(file_t)
        data_t = load_pickle(file_t)
        for u_id in range(len(data_t["uid"])):
            uid_t = data_t["uid"][u_id]
            a_len = len(data_t["a_item"][u_id])
            b_len = len(data_t["b_item"][u_id])
            if a_len > 4:
                a_all.append(uid_t)
                if b_len > 4:
                    overlap.append(uid_t)
                    b_all.append(uid_t)
            if b_len > 4:
                b_all.append(uid_t)
    print("a domain", len(a_all))
    print("b domain", len(b_all))
    print("overlap", len(overlap))
    save_pickle(a_all, "a_all.pickle")
    save_pickle(b_all, "b_all.pickle")
    save_pickle(overlap, "overlap.pickle")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1, help="file number.")
    parser.add_argument("--rate", type=float, default=0.1, help="overlapping rate.")
    args = parser.parse_args()

    # 1. Remove consecutive repeating items, e.g. 1,2,3,3,3,5,6, --> 1,2,3,5,6
    # current dataset should already get rid of this problem.
    # 2. user id, from 1-N
    # 3. item id in each domain, 1-M
    in_path = "/data/ceph/seqrec/data/business/kdd"         # original data
    out_path = "/data/ceph/seqrec/data/business/kdd_2/"     # processed data
    remove_repeats(in_path, out_path)

    # generate cross-domain scenario.
    file_path = "/data/ceph/seqrec/data/business/kdd_2"     # processed data
    overlap_ratio = args.rate
    out_path = "/data/ceph/seqrec/data/business/kdd_%d" % int(overlap_ratio * 100)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    for i in range(19):  # totally 19 files. TODO: go parallel.
        generate_data(file_path, overlap_ratio, out_path, i)
        # check the real overlapping rate.
        # statistic(file_path)
