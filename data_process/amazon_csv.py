# -*- coding: UTF-8 -*-
"""
Methods to processing the amazon review dataset.
"""
import _pickle as pickle
import gc
import gzip
import json
import os

import numpy as np
import pandas as pd


def load_pickle(filename):
    return pickle.load(open(filename, "rb"))


def parse(path_tmp):
    g = gzip.open(path_tmp, 'rb')
    for line in g:
        yield json.loads(line)


def getDF(path_tmp):
    i = 0
    df = {}
    for d in parse(path_tmp):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def generate_k_core(df, k):
    # on users
    # tmp1 = df.groupby(['reviewerID'], as_index=False)['asin'].count()
    # tmp1.rename(columns={'asin': 'cnt_item'}, inplace=True)
    # on items
    tmp2 = df.groupby(['asin'], as_index=False)['reviewerID'].count()
    tmp2.rename(columns={'reviewerID': 'cnt_user'}, inplace=True)

    df = df.merge(tmp2, on=['asin'])
    query = "cnt_user >= %d" % k
    df = df.query(query).reset_index(drop=True).copy()
    df.drop(['cnt_user'], axis=1, inplace=True)
    del tmp2
    gc.collect()
    return df


def info(data):
    """ num of user, item, max/min uid/itemID, total interaction"""
    user = set(data['reviewerID'].tolist())
    item = set(data['asin'].tolist())
    print("number of user: ", len(user))
    print("max user ID: ", max(user))
    print("Min user ID: ", min(user))
    print("number of Item: ", len(item))
    print("Max item ID: ", max(item))
    print("Min item ID: ", min(item))
    print("Interactions: ", len(data))


def user_limit(df, k=5):
    # select users with more than (include) k interacted items
    tmp1 = df.groupby(['reviewerID'], as_index=False)['asin'].count()
    tmp1.rename(columns={'asin': 'cnt_item'}, inplace=True)
    df = df.merge(tmp1, on=['reviewerID'])
    query = "cnt_item >= %d" % k
    df = df.query(query).reset_index(drop=True).copy()
    df.drop(['cnt_item'], axis=1, inplace=True)
    del tmp1
    gc.collect()
    return df


def category(df):
    # reid users and items from 0-N
    df['reviewerID'] = pd.Categorical(df.uid).codes
    df['asin'] = pd.Categorical(df.asin).codes
    df.sort_values(['reviewerID', 'unixReviewTime'], inplace=True)
    return df


def clean(df):
    # pre-processing on the original dataset TODO: change parameters accordingly.
    print("======================Information of Original dataset:")
    info(df)
    # clean with overall verified value. And selected time span.
    df = df[df.overall >= 3]                 # select positive interactions.
    df = df[df.verified]
    df = df[df.unixReviewTime > 1506816000]  # interactions after 2017, 10, 1, 0, 0, 0
    # earliest --> 2016, 10, 1, 0, 0, 0; 1506816000 --> 2017, 10, 1, 0, 0, 0

    # drop redundant columns
    df = df.drop(["overall", "verified", 'reviewerName', 'reviewText', 'summary', 'vote', 'style', 'image'], axis=1)
    print("======================Information of cleaned dataset:")
    info(df)

    # k-core
    df = generate_k_core(df, k=5)  # item
    print("======================Information of %d-core dataset:" % 5)
    info(df)

    df = user_limit(df, k=5)   # user
    print("======================Information of dataset where all users have at least %d interaction:" % 5)
    info(df)

    # Convert User_ID and Item_ID and sort by reviewerID and time
    # df['reviewerID'] = pd.Categorical(df.reviewerID).codes
    # df['reviewerID'] = df["reviewerID"] + 1
    df['asin'] = pd.Categorical(df.asin).codes
    df['asin'] = df["asin"] + 1
    df.sort_values(by=['reviewerID', "unixReviewTime"], inplace=True)

    df.head()
    print("======================final single-domain data convert to categorical data")
    info(df)
    # to list
    item_list = df.groupby('reviewerID')['asin'].apply(list).reset_index(name='asin_list')
    # ts_list = df.groupby('reviewerID')['unixReviewTime'].apply(list).reset_index(name='ts_list')

    return item_list


def main_process(source_file, outfile):
    a_df = getDF(source_file)
    a_df_cleaned_list = clean(a_df)
    print(a_df_cleaned_list.head())
    a_df_cleaned_list.to_csv(outfile, index=False, header=True)
    # generate_tfrecord(a_df_cleaned_list, out_tf, in_type="df")


def find_overlap(df1, df2):
    a_users = df1['reviewerID'].values
    b_users = df2['reviewerID'].values
    overlap_users = []
    for user in a_users:
        if user in b_users:
            overlap_users.append(user)
    print(len(a_users))
    print(len(b_users))
    print(len(overlap_users))
    print("saving overlap data")
    overlap_data = {}
    for index, data_line in df1.iterrows():
        if data_line["reviewerID"] in overlap_users:
            overlap_data[data_line["reviewerID"]] = [data_line["asin_list"]]

    for index, data_line in df2.iterrows():
        if data_line["reviewerID"] in overlap_users:
            if data_line["reviewerID"] in overlap_data:
                overlap_data[data_line["reviewerID"]].append(data_line["asin_list"])
            else:
                print("wrong user")
    return overlap_data


def statistic_pub(df1, df2):
    a_users = df1['reviewerID'].values
    b_users = df2['reviewerID'].values
    overlap_users = []
    a_only = []
    b_only = []
    for user in a_users:
        if user in b_users:
            overlap_users.append(user)
        else:
            a_only.append(user)
    for user in b_users:
        if user not in overlap_users:
            b_only.append(user)
    print(f"number of users in domain a: {len(a_users)}")
    print(len(b_users))
    print(len(a_only))
    print(len(b_only))
    print(len(overlap_users))
    print("saving overlap data")
    # a_only: users with interactions only in a domain
    # b_only: users with interactions only in b domain
    # x_user: overlapped users.
    data_info = {"a_only": a_only, "b_only": b_only, "x_user": overlap_users}

    return data_info


def str2int(in_str):
    """
    input sample: '[11506, 10463, 34296, 15541]'
    """
    data_list = in_str.strip()[1:-1].split(",")  # remove "[]" and split
    data = list(map(int, data_list))
    return data


def pf2pickle_over(overlap_dict, outfile):
    """
    save overlapped data.
    """
    data_pickle = {"seq_a": [], "len_a": [], "val_a": [], "test_a": [],
                   "seq_b": [], "len_b": [], "val_b": [], "test_b": []}
    users = list(overlap_dict.keys())
    num_recorded = 0
    for user in users:
        a_behavior, b_behavior = overlap_dict[user]
        a_behavior = str2int(a_behavior)
        b_behavior = str2int(b_behavior)

        # if len(a_behavior) < 5 or len(b_behavior) < 5:
        #   continue

        val = a_behavior[-2]
        test = a_behavior[-1]
        behavior = a_behavior[:-2]

        val_b = b_behavior[-2]
        test_b = b_behavior[-1]
        behavior_b = b_behavior[:-2]

        num_recorded += 1
        data_pickle["seq_a"].append(behavior)
        data_pickle["len_a"].append(len(behavior))
        data_pickle["val_a"].append(val)
        data_pickle["test_a"].append(test)

        data_pickle["seq_b"].append(behavior_b)
        data_pickle["len_b"].append(len(behavior_b))
        data_pickle["val_b"].append(val_b)
        data_pickle["test_b"].append(test_b)

    print("save number of samples: ", num_recorded)
    with open(outfile, "wb") as fid:
        pickle.dump(data_pickle, fid, -1)


def df2pickle(infile, outfile, in_type):
    """
    train, valid, and test split and store all results into pickle files.
    """
    if in_type == "pickle":
        data = pickle.load(open(infile, "rb"))
    elif in_type == "txt":
        data = open(infile, "r")
    elif in_type == "df":
        # data = pd.read_csv(infile)  # if input a filename
        # data = data.iterrows()
        data = infile.iterrows()
    else:
        data = infile
    # result pickle file
    data_pickle = {"seq": [], "len": [], "val": [], "test": []}

    for data_line in data:
        if in_type == "txt":
            line_splits = data_line.strip().split("|")
            user, behavior, timestamp = line_splits
            behavior = behavior.split(",")
            behavior = list(map(int, behavior))
        elif in_type == "pickle":  # pickle data
            user, behavior, timestamp = data_line[0], data_line[1], data_line[2]
        elif in_type == "df" or in_type == "csv":
            user, behavior, timestamp = data_line[1]["reviewerID"], data_line[1]["asin_list"], None
            if type(behavior) == str:
                behavior = [int(val) for val in behavior[1:-1].strip().split(",")]
        else:
            behavior = [int(val) for val in data_line[1:-1].strip().split(",")]
        # if len(behavior) < 3:  # already 5-cores.
        #    continue
        val = behavior[-2]
        test = behavior[-1]
        behavior = behavior[:-2]

        data_pickle["seq"].append(behavior)
        data_pickle["len"].append(len(behavior))
        data_pickle["val"].append(val)
        data_pickle["test"].append(test)

    print("saving", outfile)
    with open(outfile, "wb") as fid:
        pickle.dump(data_pickle, fid, -1)


def cross_data(df1, df2, data_info, a_name, b_name, path):
    """
    Args:
        df1: dataframe in domain a
        df2: dataframe in domain b
        data_info: which {"a_only": [], "b_only": [], "x_users": []}
        a_name: name of domain a
        b_name: name of domain b
        path: out path.
    Returns:
        a_name.pickle
        a_name_only.pickle
        b_name.pickle
        b_name_only.pickle
        "a_name"_"b_name".pickle
    """
    # Out put config
    a_all_name = os.path.join(path, a_name + ".pickle")
    b_all_name = os.path.join(path, b_name + ".pickle")
    a_only_name = os.path.join(path, a_name + "_only.pickle")
    b_only_name = os.path.join(path, b_name + "_only.pickle")
    overlap_name = os.path.join(path, a_name + "_" + b_name + ".pickle")

    # all users in each domain.
    df2pickle(df1, a_all_name, in_type="df")
    df2pickle(df2, b_all_name, in_type="df")

    # overlap data
    overlap_users = data_info["x_user"]
    overlap_data = {}
    for _, data_line in df1.iterrows():
        if data_line["reviewerID"] in overlap_users:
            overlap_data[data_line["reviewerID"]] = [data_line["asin_list"]]

    for _, data_line in df2.iterrows():
        if data_line["reviewerID"] in overlap_users:
            if data_line["reviewerID"] in overlap_data:
                overlap_data[data_line["reviewerID"]].append(data_line["asin_list"])
            else:
                print("wrong user")
    pf2pickle_over(overlap_data, overlap_name)

    # domain-specific users
    a_only_data = find_domain_only_data(df1, overlap_users)
    b_only_data = find_domain_only_data(df2, overlap_users)

    df2pickle(a_only_data, a_only_name, in_type="list")
    df2pickle(b_only_data, b_only_name, in_type="list")


def find_domain_only_data(df, overlap_user):
    data = []
    for _, d_row in df.iterrows():
        if d_row["reviewerID"] not in overlap_user:
            data.append(d_row["asin_list"])
    return data


def data_frequency(infile, outfile, total=None):
    data = load_pickle(infile)
    freq = np.zeros(total, dtype=float)
    for index in range(len(data["val"])):
        seq = data["seq"][index]
        val = data["val"][index]
        test = data["test"][index]
        for item in seq + [val] + [test]:
            freq[item - 1] += 1
    # check if there are items with 0 frequency. (which is incorrect.)
    zero_index = np.where(freq == 0)
    if len(zero_index[0]) > 0:
        print(zero_index)
    else:
        # add 0 at the beginning, for pad_index.
        freq = np.insert(freq, 0, 0, axis=0)
        with open(outfile, "wb") as fid:
            pickle.dump(freq, fid, -1)


def freq_main():
    names = ["book", "movie", "sport", "cloth"]
    # in
    movie = "/data/ceph/seqrec/data/public/Amazon_torch/movie_book/movie.pickle"
    book = "/data/ceph/seqrec/data/public/Amazon_torch/movie_book/book.pickle"
    clothing = "/data/ceph/seqrec/data/public/Amazon_torch/sport_cloth/cloth.pickle"
    sport = "/data/ceph/seqrec/data/public/Amazon_torch/sport_cloth/sport.pickle"
    # out
    movie_o = "/data/ceph/seqrec/data/public/Amazon_torch/movie_book/movie_freq.pickle"
    book_o = "/data/ceph/seqrec/data/public/Amazon_torch/movie_book/book_freq.pickle"
    clothing_o = "/data/ceph/seqrec/data/public/Amazon_torch/sport_cloth/cloth_freq.pickle"
    sport_o = "/data/ceph/seqrec/data/public/Amazon_torch/sport_cloth/sport_freq.pickle"
    org_data = {"book": book, "movie": movie, "sport": sport, "cloth": clothing}
    output_data = {"book": book_o, "movie": movie_o, "sport": sport_o, "cloth": clothing_o}
    # total_all = [53014, 5637, 12790, 44512]    # [1, n]
    total_all = [51366, 5536, 11835, 42139]      # [1, n]
    for idx in range(len(names)):
        data_frequency(org_data[names[idx]], output_data[names[idx]], total=total_all[idx])


def form_cross_domain_sets(file_a, file_b, a_name, b_name, out_path):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    a_data = pd.read_csv(file_a)
    b_data = pd.read_csv(file_b)
    data_info = statistic_pub(a_data, b_data)
    # print(len(data_info[0]))
    cross_data(a_data, b_data, data_info, a_name, b_name, out_path, rate=0)


if __name__ == "__main__":
    '''                preprocessing of the Amazon datasets                '''
    # download the following datasets from: https://nijianmo.github.io/amazon/index.html
    data_dir = "/Users/chain/git/Recommendation/Amazon_data"
    out_dir = "/Users/chain/git/Recommendation/Amazon_data/guru"
    movie_source = os.path.join(data_dir, "Movies_and_TV_5.json.gz")
    sport_source = os.path.join(data_dir, "Movies_and_TV_5.json.gz")

    # preprocessed dataset
    movie_out = os.path.join(out_dir, "movie.csv")
    sport_out = os.path.join(out_dir, "sport.csv")
    # main_process(movie_source, movie_out)
    # main_process(sport_source, sport_out)

    # form cross-domain datasets
    form_cross_domain_sets(movie_out, sport_out, 'movie', 'sport',
                           out_path=os.path.join(out_dir, "movie_sport"))
    # calculate item frequency in each domain.
    # freq_main()
