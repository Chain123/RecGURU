# -*- coding: UTF-8 -*-
"""
NOT IN USE, see amazon_csv.py and business_process.py
"""

import os
import _pickle as pickle


def load_pickle(filename):
    f = open(filename, "rb")
    return pickle.load(f)  # encoding="latin"


def save_pickle(data, filename):
    with open(filename, "wb") as fid:
        pickle.dump(data, fid, -1)


def item_frequency(files, out_name):
    item_freq_dict = {}
    for file in files:
        data = load_pickle(file)
        for index in range(len(data['val'])):
            # train item
            for item in data['seq']:
                if item not in item_freq_dict:
                    item_freq_dict[item] = 0
                item_freq_dict[item] += 1
            # valid item
            if data['val'] not in item_freq_dict:
                item_freq_dict[data['val']] = 0
            item_freq_dict[data['val']] += 1
            # test item
            if data['test'] not in item_freq_dict:
                item_freq_dict[data['test']] = 0
            item_freq_dict[data['test']] += 1
    save_pickle(item_freq_dict, out_name)


if __name__ == "__main__":
    # amazon dataset
    domain = "movie"
    data_dir = "/Users/chain/git/Recommendation/Amazon_data/guru/movie_sport"
    file_list = [os.path.join(data_dir, val) for val in os.listdir(data_dir)
                 if "_" not in val and domain in val]  # domain data.
    out_path = os.path.join(data_dir, f"{domain}_item_freq.pickle")
    item_frequency(file_list, out_path)
    # collected dataset. See business_process.py
