# -*- coding: UTF-8 -*-
"""
Data loader:
"""
import torch.utils.data as data
import torch
import _pickle as pickle
import numpy as np
from torch.utils.data.dataloader import default_collate
from torch import FloatTensor as FT
import copy
import random


def load_pickle(filename):
    f = open(filename, "rb")
    return pickle.load(f)  # encoding="latin"


def save_pickle(dict_name, file_name):
    with open(file_name, "wb") as fid:
        pickle.dump(dict_name, fid, -1)


def seq_padding(seq, length_enc, len_dec, eos):
    seq = list(seq)
    if len(seq) >= length_enc:
        enc_in = seq[-length_enc + 1:] + [eos]
    else:
        enc_in = [0] * (length_enc - len(seq) - 1) + seq + [eos]

    dec_in = [0, 0] + enc_in[0:-2]
    dec_out = [0, 0] + enc_in[1:-1]
    dec_in = dec_in[-len_dec:]
    dec_out = dec_out[-len_dec:]
    return np.array(enc_in), np.array(dec_in), np.array(dec_out)


def test_seq_gen(seq, length_enc, len_dec, eos, val):
    seq = list(seq)
    seq_test = seq + [val]
    if len(seq) >= length_enc:
        eval_enc_in = seq[-length_enc + 1:] + [eos]
    else:
        eval_enc_in = [0] * (length_enc - len(seq) - 1) + seq + [eos]
    dec_in = [0] + eval_enc_in[0:-1]  # remove the eos
    dec_in = dec_in[-len_dec:]

    if len(seq_test) >= length_enc:
        test_enc_in = seq_test[-length_enc + 1:] + [eos]
    else:
        test_enc_in = [0] * (length_enc - len(seq_test) - 1) + seq_test + [eos]
    t_dec_in = [0] + test_enc_in[0:-1]
    t_dec_in = t_dec_in[-len_dec:]
    return np.array(eval_enc_in), np.array(dec_in), np.array(test_enc_in), np.array(t_dec_in)


class pickle_loader_eval(data.Dataset):
    """
    Data loader for sequential recommendation;
    pickle file contains: "seq", "len", "val", "test".
    """

    def __init__(self, filename, param, num_n, seq_len=None, wf=None, domain="a"):
        self.seq = []  # need padding.
        self.val = []
        self.test = []
        self.seq_len = seq_len
        self.num_n = num_n
        self.enc_len = param.enc_maxlen
        self.dec_len = param.rec_maxlen
        if domain == "a":
            self.eos = param.vocab_size_a
        else:
            self.eos = param.vocab_size_b
        self.param = param
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None

        for file in filename:
            # print(file)
            data_tmp = load_pickle(file)
            if self.seq_len:  # select user with behavior length fall in range "seq_len"
                for i in range(len(data_tmp["val"])):
                    if self.seq_len[1] > len(data_tmp["seq"][i]) > self.seq_len[0]:
                        self.seq.append(data_tmp["seq"][i])
                        self.val.append(data_tmp["val"][i])
                        self.test.append(data_tmp["test"][i])
            else:
                self.seq.extend(data_tmp["seq"])
                self.val.extend(data_tmp["val"])
                self.test.extend(data_tmp["test"])

        # self.seq = np.array(self.seq)
        # self.val = np.array(self.val)
        # self.test = np.array(self.test)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        seq, val, test = self.seq[index], self.val[index], self.test[index]
        eval_enc_in, dec_in, test_enc_in, t_dec_in = test_seq_gen(seq, self.enc_len, self.dec_len, self.eos, val)
        weights_f = copy.deepcopy(self.weights)
        for val in seq:
            weights_f[seq] = 0
        weights_f[val] = 0
        weights_f[test] = 0

        weights_r = np.ones(len(self.weights))
        weights_r[0] = 0
        for val in seq:
            weights_r[seq] = 0
        weights_r[val] = 0
        weights_r[test] = 0
        weights_r = torch.from_numpy(weights_r)

        n_items_f = torch.multinomial(weights_f,
                                      self.param.candidate_size,
                                      replacement=True)
        n_items_r = torch.multinomial(weights_r,
                                      self.param.candidate_size,
                                      replacement=True)
        # n_items_r = FT(self.param.candidate_size).uniform_(0, self.param.vocab_size - 1).long()

        return (eval_enc_in, dec_in, val), (test_enc_in, t_dec_in, test), n_items_f, n_items_r

    def __len__(self):
        return len(self.val)


class pickle_loader_eval_tencent(data.Dataset):
    """
    Data loader for sequential recommendation;
    pickle file contains: "seq", "len", "val", "test".
    """

    def __init__(self, filename, param, num_n, seq_len=None, domain="a", wf=None):
        self.seq = []  # need padding.
        self.val = []
        self.test = []
        self.seq_len = seq_len
        self.num_n = num_n
        self.enc_len = param.enc_maxlen
        self.dec_len = param.rec_maxlen
        if domain == "a":
            self.eos = param.vocab_size_a
        else:
            self.eos = param.vocab_size_b
        self.param = param

        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None

        for file in filename:
            # print(file)
            data_tmp = load_pickle(file)
            if "over" in file:
                self.seq.extend(data_tmp["seq_%s" % domain])
                self.val.extend(data_tmp["val_%s" % domain])
                self.test.extend(data_tmp["test_%s" % domain])
            else:
                if self.seq_len:  # select user with behavior length fall in range "seq_len"
                    for i in range(len(data_tmp["val"])):
                        if self.seq_len[1] > len(data_tmp["seq"][i]) > self.seq_len[0]:
                            self.seq.append(data_tmp["seq"][i])
                            self.val.append(data_tmp["val"][i])
                            self.test.append(data_tmp["test"][i])
                else:
                    self.seq.extend(data_tmp["seq"])
                    self.val.extend(data_tmp["val"])
                    self.test.extend(data_tmp["test"])

        # self.seq = np.array(self.seq)
        # self.val = np.array(self.val)
        # self.test = np.array(self.test)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        seq, val, test = self.seq[index], self.val[index], self.test[index]
        eval_enc_in, dec_in, test_enc_in, t_dec_in = test_seq_gen(seq, self.enc_len, self.dec_len, self.eos, val)
        if self.weights is not None:
            weights_f = copy.deepcopy(self.weights)
            for val in seq:
                weights_f[seq] = 0
            weights_f[val] = 0
            weights_f[test] = 0
            n_items_f = torch.multinomial(weights_f,
                                          self.param.candidate_size,
                                          replacement=True)
        else:
            n_items_f = 1
        weights_r = np.ones(self.eos)
        weights_r[0] = 0
        for val in seq:
            weights_r[seq] = 0
        weights_r[val] = 0
        weights_r[test] = 0
        weights_r = torch.from_numpy(weights_r)

        n_items_r = torch.multinomial(weights_r,
                                      self.param.candidate_size,
                                      replacement=True)

        return (eval_enc_in, dec_in, val), (test_enc_in, t_dec_in, test), n_items_f, n_items_r

    def __len__(self):
        return len(self.val)


class pickle_loader(data.Dataset):
    """
    Data loader for sequential recommendation;
    pickle file contains: "seq", "len", "val", "test".
    """

    def __init__(self, filename, param, num_n, seq_len=None, wf=None, eval_n=False, rec=False, domain="a"):
        self.seq = []  # need padding.
        self.val = []
        self.test = []
        self.seq_len = seq_len  # a range [shortest, longest]
        self.num_n = num_n
        self.enc_len = param.enc_maxlen
        self.domain = domain
        if rec:
            self.dec_len = param.rec_maxlen
        else:
            self.dec_len = param.enc_maxlen
        if domain == "a":
            self.eos = param.vocab_size_a
        else:
            self.eos = param.vocab_size_b
        self.param = param
        self.eval = eval_n
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None

        for file in filename:
            # print(file)
            data_tmp = load_pickle(file)
            if self.seq_len:  # select user with behavior length fall in range "seq_len"
                for i in range(len(data_tmp["val"])):
                    if self.seq_len[1] > len(data_tmp["seq"][i]) > self.seq_len[0]:
                        self.seq.append(data_tmp["seq"][i])
                        self.val.append(data_tmp["val"][i])
                        self.test.append(data_tmp["test"][i])
            else:
                self.seq.extend(data_tmp["seq"])
                self.val.extend(data_tmp["val"])
                self.test.extend(data_tmp["test"])
        # self.seq = np.array(self.seq)
        # self.val = np.array(self.val)
        # self.test = np.array(self.test)
        # print(np.max(self.seq), self.seq.shape, self.seq[0])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        seq, val, test = self.seq[index], self.val[index], self.test[index]
        # pad: can also be achieved by Transformer. (pre-processing)
        enc_in, dec_in, dec_out = seq_padding(seq, self.enc_len, self.dec_len, self.eos)
        if self.eval:
            num_n = self.param.candidate_size
        else:
            num_n = self.param.enc_maxlen * self.num_n
        if self.weights is not None:
            weights = copy.deepcopy(self.weights)
            for val in seq:
                weights[seq] = 0
            weights[val] = 0
            weights[test] = 0
            # n_items = torch.multinomial(weights,
            #                             num_n,
            #                             replacement=True)
        else:
            if self.domain == "a":
                v_size = self.param.vocab_size_a
            else:
                v_size = self.param.vocab_size_b
            weights = np.ones(v_size)
            weights[0] = 0
            for val in seq:
                weights[seq] = 0
            weights[val] = 0
            weights[test] = 0
            weights = torch.from_numpy(weights)
            # n_items = FT(num_n).uniform_(0, self.param.vocab_size - 1).long()
        n_items = torch.multinomial(weights,
                                    num_n,
                                    replacement=True)
        # TODO: generate seq for enc_input, dec_input and dec_output.
        return (enc_in, dec_in, dec_out), n_items, val, test

    def __len__(self):
        return len(self.val)


class pickle_loader_tencent(data.Dataset):
    """
    Data loader for sequential recommendation;
    pickle file contains: "seq", "len", "val", "test".
    """

    def __init__(self, filename, param, num_n, seq_len=None, eval_n=False, rec=False, domain="a", wf=None):
        self.seq = []  # need padding.
        self.val = []
        self.test = []
        self.seq_len = seq_len  # a range [shortest, longest]
        self.num_n = num_n
        self.enc_len = param.enc_maxlen
        self.domain = domain
        if rec:
            self.dec_len = param.rec_maxlen
        else:
            self.dec_len = param.enc_maxlen
        if domain == "a":
            self.eos = param.vocab_size_a
        else:
            self.eos = param.vocab_size_b

        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None

        self.param = param
        self.eval = eval_n
        for file in filename:
            # print(file)
            data_tmp = load_pickle(file)
            if "over" in file:
                self.seq.extend(data_tmp["seq_%s" % domain])
                self.val.extend(data_tmp["val_%s" % domain])
                self.test.extend(data_tmp["test_%s" % domain])
            else:
                if self.seq_len:  # select user with behavior length fall in range "seq_len"
                    for i in range(len(data_tmp["val"])):
                        if self.seq_len[1] > len(data_tmp["seq"][i]) > self.seq_len[0]:
                            self.seq.append(data_tmp["seq"][i])
                            self.val.append(data_tmp["val"][i])
                            self.test.append(data_tmp["test"][i])
                else:
                    self.seq.extend(data_tmp["seq"])
                    self.val.extend(data_tmp["val"])
                    self.test.extend(data_tmp["test"])
        self.seq = np.array(self.seq)
        self.val = np.array(self.val)
        self.test = np.array(self.test)
        print(np.max(self.seq))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        seq, val, test = self.seq[index], self.val[index], self.test[index]
        # pad: can also be achieved by Transformer. (pre-processing)
        enc_in, dec_in, dec_out = seq_padding(seq, self.enc_len, self.dec_len, self.eos)
        if self.eval:
            num_n = self.param.candidate_size
        else:
            num_n = self.param.enc_maxlen * self.num_n

        if self.domain == "a":
            v_size = self.param.vocab_size_a
        else:
            v_size = self.param.vocab_size_b
        if self.weights is not None:
            weights = copy.deepcopy(self.weights)
        else:
            weights = np.ones(v_size)
            weights[0] = 0

        for val in seq:
            weights[seq] = 0
        weights[val] = 0
        weights[test] = 0
        if self.weights is None:
            weights = torch.from_numpy(weights)
        n_items = torch.multinomial(weights,
                                    num_n,
                                    replacement=True)
        # TODO: generate seq for enc_input, dec_input and dec_output.
        return (enc_in, dec_in, dec_out), n_items, val, test

    def __len__(self):
        return len(self.val)


def my_collate(batch):
    # print(batch)
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def dataloader_gen_tencent(all_files, param, num_n, train=True, seq_len=None,
                           rec=False, domain="a", n_w=4, wf=None):
    """
    Normal dataloader
    :param n_w:             number of workers
    :param domain:
    :param wf:
    :param rec:             rec=True, determine the seq len to the rec model. (dec_in dec_out)
    :param param:
    :param num_n:
    :param seq_len:         selected users with a specific behavior length fall in the range of seq_len = [l, h]
    :param all_files: file list of pickle files
    :param train: bool variable, output train a data loader?
    :return: data loader

    consume the returned dataloader with tqdm:
    for i, (seq, val, test) in enumerate(tqdm(dataloader)):
    """
    if not train:
        dataset = pickle_loader_eval_tencent(all_files, param, num_n, seq_len=seq_len, domain=domain, wf=wf)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size_val,
                                                 shuffle=train, num_workers=n_w,
                                                 drop_last=True, collate_fn=my_collate)
    else:
        dataset = pickle_loader_tencent(all_files, param, num_n, seq_len=seq_len, rec=rec, domain=domain, wf=wf)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size,
                                                 shuffle=train, num_workers=n_w,
                                                 drop_last=True, collate_fn=my_collate)
    return dataloader


def dataloader_gen(all_files, param, num_n, train=True, seq_len=None,
                   wf=None, rec=False, domain="a", n_w=4):
    """
    Normal dataloader
    :param n_w:             number of workers
    :param domain:
    :param wf:
    :param rec:             rec=True, determine the seq len to the rec model. (dec_in dec_out)
    :param param:
    :param num_n:
    :param seq_len:         selected users with a specific behavior length fall in the range of seq_len = [l, h]
    :param all_files: file list of pickle files
    :param train: bool variable, output train a data loader?
    :return: data loader

    consume the returned dataloader with tqdm:
    for i, (seq, val, test) in enumerate(tqdm(dataloader)):
    """
    if not train:
        dataset = pickle_loader_eval(all_files, param, num_n, seq_len=seq_len, wf=wf, domain=domain)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size_val,
                                                 shuffle=train, num_workers=n_w,
                                                 drop_last=True, collate_fn=my_collate)
    else:
        dataset = pickle_loader(all_files, param, num_n, seq_len=seq_len, wf=wf, rec=rec, domain=domain)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size,
                                                 shuffle=train, num_workers=n_w,
                                                 drop_last=True, collate_fn=my_collate)
    return dataloader


class pickle_loader_over(data.Dataset):
    """
    Data loader for sequential recommendation;
    pickle file contains: "seq", "len", "val", "test".
    """

    def __init__(self, filename, param):
        self.seq_a = []  # need padding.
        self.val_a = []
        self.test_a = []
        self.seq_b = []  # need padding.
        self.val_b = []
        self.test_b = []
        self.param = param

        for file_t in filename:
            # print(file_t)
            data_tmp = load_pickle(file_t)
            self.seq_a.extend(data_tmp["seq_a"])
            self.val_a.extend(data_tmp["val_a"])
            self.test_a.extend(data_tmp["test_a"])

            self.seq_b.extend(data_tmp["seq_b"])
            self.val_b.extend(data_tmp["val_b"])
            self.test_b.extend(data_tmp["test_b"])

        # self.seq_a = np.array(self.seq_a)
        # self.val_a = np.array(self.val_a)
        # self.test_a = np.array(self.test_a)
        #
        # self.seq_b = np.array(self.seq_b)
        # self.val_b = np.array(self.val_b)
        # self.test_b = np.array(self.test_b)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        seq_a, val_a, test_a = self.seq_a[index], self.val_a[index], self.test_a[index]
        seq_b, val_b, test_b = self.seq_b[index], self.val_b[index], self.test_b[index]
        enc_in_a, dec_in_a, dec_out_a = seq_padding(seq_a,
                                                    self.param.rec_maxlen,
                                                    self.param.enc_maxlen,
                                                    self.param.vocab_size_a)
        enc_in_b, dec_in_b, dec_out_b = seq_padding(seq_b,
                                                    self.param.rec_maxlen,
                                                    self.param.enc_maxlen,
                                                    self.param.vocab_size_b)
        n_items_a = 1
        n_items_b = 1
        return (enc_in_a, dec_in_a, dec_out_a, val_a, test_a, n_items_a), \
               (enc_in_b, dec_in_b, dec_out_b, val_b, test_b, n_items_b)

    def __len__(self):
        return len(self.val_a)


# class pickle_loader_over_tencent(data.Dataset):
#     """
#     Data loader for sequential recommendation;
#     pickle file contains: "seq", "len", "val", "test".
#     """
#
#     def __init__(self, filename, param):
#         self.seq_a = []  # need padding.
#         self.val_a = []
#         self.test_a = []
#         self.seq_b = []  # need padding.
#         self.val_b = []
#         self.test_b = []
#         self.param = param
#
#         for file_t in filename:
#             print(file_t)
#             data_tmp = load_pickle(file_t)
#             self.seq_a.extend(data_tmp["seq_a"])
#             self.val_a.extend(data_tmp["val_a"])
#             self.test_a.extend(data_tmp["test_a"])
#
#             self.seq_b.extend(data_tmp["seq_b"])
#             self.val_b.extend(data_tmp["val_b"])
#             self.test_b.extend(data_tmp["test_b"])
#         self.seq_a = np.array(self.seq_a)
#         self.val_a = np.array(self.val_a)
#         self.test_a = np.array(self.test_a)
#
#         self.seq_b = np.array(self.seq_b)
#         self.val_b = np.array(self.val_b)
#         self.test_b = np.array(self.test_b)
#         # print(np.max(self.seq))
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         seq_a, val_a, test_a = self.seq_a[index], self.val_a[index], self.test_a[index]
#         seq_b, val_b, test_b = self.seq_b[index], self.val_b[index], self.test_b[index]
#         enc_in_a, dec_in_a, dec_out_a = seq_padding(seq_a,
#                                                     self.param.rec_maxlen,
#                                                     self.param.enc_maxlen,
#                                                     self.param.vocab_size_a)
#         enc_in_b, dec_in_b, dec_out_b = seq_padding(seq_b,
#                                                     self.param.rec_maxlen,
#                                                     self.param.enc_maxlen,
#                                                     self.param.vocab_size_b)
#         n_items_a = 1
#         n_items_b = 1
#         return (enc_in_a, dec_in_a, dec_out_a, val_a, test_a, n_items_a), \
#                (enc_in_b, dec_in_b, dec_out_b, val_b, test_b, n_items_b)
#
#     def __len__(self):
#         return len(self.val_a)


def overlap_split(file_t):
    data_tmp = load_pickle(file_t)
    tmp = list(zip(data_tmp["seq_a"], data_tmp["val_a"], data_tmp["test_a"],
                   data_tmp["seq_b"], data_tmp["val_b"], data_tmp["test_b"]))
    random.shuffle(tmp)
    seq_a, val_a, test_a, seq_b, val_b, test_b = zip(*tmp)
    return seq_a, val_a, test_a, seq_b, val_b, test_b


class pickle_loader_over_em(data.Dataset):
    """
    Data loader for sequential recommendation;
    pickle file contains: "seq", "len", "val", "test".
    """

    def __init__(self, all_data, param, wf=None, domain="a", recom=False, test_=False):
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None
        self.recom = recom
        self.param = param
        self.seq_a = np.array(all_data[0])
        self.val_a = np.array(all_data[1])
        self.test_a = np.array(all_data[2])

        self.seq_b = np.array(all_data[3])
        self.val_b = np.array(all_data[4])
        self.test_b = np.array(all_data[5])
        if domain == "a":
            self.a_size = self.param.vocab_size_a
            self.b_size = self.param.vocab_size_b
        else:
            self.a_size = self.param.vocab_size_b
            self.b_size = self.param.vocab_size_a
        self.domain = domain
        self.test_ = test_

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        seq_a, val_a, test_a = self.seq_a[index], self.val_a[index], self.test_a[index]
        seq_b, val_b, test_b = self.seq_b[index], self.val_b[index], self.test_b[index]
        if self.test_:
            eval_enc_in_a, dec_in_a, test_enc_in_a, t_dec_in_a = test_seq_gen(seq_a,
                                                                              self.param.enc_maxlen,
                                                                              self.param.rec_maxlen,
                                                                              self.param.vocab_size_a,
                                                                              val_a)
            eval_enc_in_b, dec_in_b, test_enc_in_b, t_dec_in_b = test_seq_gen(seq_b,
                                                                              self.param.enc_maxlen,
                                                                              self.param.rec_maxlen,
                                                                              self.param.vocab_size_b,
                                                                              val_b)
            if self.recom:
                weights = np.ones(self.param.vocab_size)  # target seq
                weights[0] = 0
                if self.domain == "a":
                    seq = seq_a
                    weights[val_a] = 0
                    weights[test_a] = 0
                else:
                    seq = seq_b
                    weights[val_b] = 0
                    weights[test_b] = 0
                for val in seq:
                    weights[val] = 0
                weights = torch.from_numpy(weights)
            else:
                weights = None

            num = self.param.candidate_size
            n_items_a = torch.multinomial(weights,
                                          num,
                                          replacement=True)
            return (eval_enc_in_a, dec_in_a, val_a, test_enc_in_a, t_dec_in_a, test_a, n_items_a), \
                   (eval_enc_in_b, dec_in_b, test_enc_in_b, t_dec_in_b)

        else:
            enc_in_a, dec_in_a, dec_out_a = seq_padding(seq_a,
                                                        self.param.rec_maxlen,
                                                        self.param.enc_maxlen,
                                                        self.a_size)
            enc_in_b, dec_in_b, dec_out_b = seq_padding(seq_b,
                                                        self.param.rec_maxlen,
                                                        self.param.enc_maxlen,
                                                        self.b_size)
            if self.recom:
                weights = np.ones(self.param.vocab_size)  # target seq
                weights[0] = 0
                if self.domain == "a":
                    seq = seq_a
                    weights[val_a] = 0
                    weights[test_a] = 0
                else:
                    seq = seq_b
                    weights[val_b] = 0
                    weights[test_b] = 0
                for val in seq:
                    weights[val] = 0
                weights = torch.from_numpy(weights)
                # n_items = FT(num_n).uniform_(0, self.param.vocab_size - 1).long()

                num = self.param.n_bpr_neg * self.param.enc_maxlen
                n_items_b = 1
                n_items_a = torch.multinomial(weights,
                                              num,
                                              replacement=True)
            else:
                n_items_a = 1
                n_items_b = 1
            return (enc_in_a, dec_in_a, dec_out_a, val_a, test_a, n_items_a), \
                   (enc_in_b, dec_in_b, dec_out_b, val_b, test_b, n_items_b)

    def __len__(self):
        return len(self.val_a)


class pickle_loader_only_em(data.Dataset):
    """
    Data loader for sequential recommendation;
    pickle file contains: "seq", "len", "val", "test".
    """

    def __init__(self, all_data, param, wf=None, domain="a", test_=False):
        self.param = param
        self.seq = np.array(all_data[0])
        self.val = np.array(all_data[1])
        self.test = np.array(all_data[2])
        self.test_ = test_
        if domain == "a":
            self.size = self.param.vocab_size_a
        else:
            self.size = self.param.vocab_size_b
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None
        self.domain = domain

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        seq, val, test = self.seq[index], self.val[index], self.test[index]
        enc_in, dec_in, dec_out = seq_padding(seq,
                                              self.param.rec_maxlen,
                                              self.param.enc_maxlen,
                                              self.size)

        if self.weights is not None:
            weights = copy.deepcopy(self.weights)
            for val in seq:
                weights[seq] = 0
            weights[val] = 0
            weights[test] = 0
        else:
            if self.domain == "a":
                v_size = self.param.vocab_size_a
            else:
                v_size = self.param.vocab_size_b
            weights = np.ones(v_size)
            weights[0] = 0
            for val in seq:
                weights[seq] = 0
            weights[val] = 0
            weights[test] = 0
            weights = torch.from_numpy(weights)
            # n_items = FT(num_n).uniform_(0, self.param.vocab_size - 1).long()
        if self.test_:
            num = self.param.candidate_size
        else:
            num = self.param.n_bpr_neg * self.param.enc_maxlen
        n_items = torch.multinomial(weights,
                                    num,
                                    replacement=True)
        return enc_in, dec_in, dec_out, val, test, n_items

    def __len__(self):
        return len(self.val)


# dataloader_gen_em_val
def dataloader_gen_em_val(files, param, n_w=2, wf=None, target="a"):
    """
    :param wf:
    :param target:
    :param files:       [overlap files, domain only file]
    :param param:       hyper parameter namespace.
    :param n_w:         number of workers.
    :return:            data loader for overlapped data.
    """
    overlap_data = load_pickle(files[0])
    if target == "a":
        overlap_train = [overlap_data["seq_a"], overlap_data["val_a"], overlap_data["test_a"],
                         overlap_data["seq_b"], overlap_data["val_b"], overlap_data["test_b"]]
    else:
        overlap_train = [overlap_data["seq_b"], overlap_data["val_b"], overlap_data["test_b"],
                         overlap_data["seq_a"], overlap_data["val_a"], overlap_data["test_a"]]
    overlap_loader = pickle_loader_over_em(overlap_train, param, wf=wf, domain=target, recom=True, test_=True)
    data_over = torch.utils.data.DataLoader(overlap_loader, batch_size=param.batch_size_over,
                                            shuffle=True, num_workers=n_w,
                                            drop_last=True, collate_fn=my_collate)

    #
    data_only = dataloader_gen([files[1]], param, param.n_bpr_neg,
                               train=False, seq_len=None, wf=wf,
                               domain=param.target_domain)

    return data_over, data_only


def dataloader_gen_over_em(files, param, n_w=2, ratio=0.2, target="a"):
    """
    :param target:
    :param ratio:
    :param files:       overlap files
    :param param:       hyper parameter namespace.
    :param n_w:         number of workers.
    :return:            data loader for overlapped data.
    """
    seq_a, val_a, test_a, seq_b, val_b, test_b = overlap_split(files)
    n_train = int(len(val_a) * (1 - ratio))
    if target == "a":
        train_data = [seq_a[0:n_train], val_a[0:n_train], test_a[0:n_train],
                      seq_b[0:n_train], val_b[0:n_train], test_b[0:n_train]]
        test_data = [seq_a[n_train:], val_a[n_train:], test_a[n_train:],
                     seq_b[n_train:], val_b[n_train:], test_b[n_train:]]
    else:
        train_data = [seq_b[0:n_train], val_b[0:n_train], test_b[0:n_train],
                      seq_a[0:n_train], val_a[0:n_train], test_a[0:n_train]]
        test_data = [seq_b[n_train:], val_b[n_train:], test_b[n_train:],
                     seq_a[n_train:], val_a[n_train:], test_a[n_train:]]

    dataset_train = pickle_loader_over_em(train_data, param, domain=target)
    dataset_test = pickle_loader_over_em(test_data, param, domain=target)
    print("====== target domain", target)
    print("maximum train a ", np.max(train_data[0]))
    print("maximum train b ", np.max(train_data[3]))

    data_train = torch.utils.data.DataLoader(dataset_train, batch_size=param.batch_size_over,
                                             shuffle=True, num_workers=n_w,
                                             drop_last=True, collate_fn=my_collate)
    data_test = torch.utils.data.DataLoader(dataset_test, batch_size=int(param.batch_size_over / 2),
                                            shuffle=False, num_workers=n_w,
                                            drop_last=True, collate_fn=my_collate)
    return data_train, data_test


def dataloader_em_train(files, param, n_w=2, wf=None, target="a"):
    """
    :param wf:
    :param target:
    :param files:       [overlap files, domain only file]
    :param param:       hyper parameter namespace.
    :param n_w:         number of workers.
    :return:            data loader for overlapped data.
    """

    overlap_data = load_pickle(files[0])
    only_data = load_pickle(files[1])
    if target == "a":
        overlap_train = [overlap_data["seq_a"], overlap_data["val_a"], overlap_data["test_a"],
                         overlap_data["seq_b"], overlap_data["val_b"], overlap_data["test_b"]]

        # only_train = [only_data["seq_a"], only_data["val_a"], only_data["test_a"],
        #               only_data["seq_b"], only_data["val_b"], only_data["test_b"]]
    else:
        overlap_train = [overlap_data["seq_b"], overlap_data["val_b"], overlap_data["test_b"],
                         overlap_data["seq_a"], overlap_data["val_a"], overlap_data["test_a"]]

        # only_train = [only_data["seq_b"], only_data["val_b"], only_data["test_b"],
        #               only_data["seq_a"], only_data["val_a"], only_data["test_a"]]
    only_train = [only_data["seq"], only_data["val"], only_data["test"]]

    overlap_loader = pickle_loader_over_em(overlap_train, param, wf=wf, domain=target, recom=True)
    only_loader = pickle_loader_only_em(only_train, param, wf=wf, domain=target)

    data_over = torch.utils.data.DataLoader(overlap_loader, batch_size=param.batch_size_over,
                                            shuffle=True, num_workers=n_w,
                                            drop_last=True, collate_fn=my_collate)

    data_only = torch.utils.data.DataLoader(only_loader, batch_size=param.batch_size,
                                            shuffle=True, num_workers=n_w,
                                            drop_last=True, collate_fn=my_collate)
    return data_over, data_only


def dataloader_gen_over(files, param, n_w=4):
    """
    :param files:       overlap files
    :param param:       hyper parameter namespace.
    :param n_w:         number of workers.
    :return:            data loader for overlapped data.
    """
    dataset = pickle_loader_over(files, param)
    # print(param.batch_size_over)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size_over,
                                             shuffle=True, num_workers=n_w,
                                             drop_last=True, collate_fn=my_collate)
    return dataloader


# def dataloader_gen_over_tencent(files, param, n_w=4):
#     """
#     :param files:       overlap files
#     :param param:       hyper parameter namespace.
#     :param n_w:         number of workers.
#     :return:            data loader for overlapped data.
#     """
#     dataset = pickle_loader_over(files, param)
#     # print(param.batch_size_over)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size_over,
#                                              shuffle=True, num_workers=n_w,
#                                              drop_last=True, collate_fn=my_collate)
#     return dataloader


class BPRData_cross_e(data.Dataset):
    def __init__(self, pairs, seqs, user, num_item, num_ng=0, wf=None, off_set=0):
        super(BPRData_cross_e, self).__init__()
        """ 
        num_item = item_a + item_b
        """
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None
        self.pair_data = pairs
        self.users = user
        self.num_item = num_item
        self.num_ng = num_ng
        self.offset = off_set
        self.seqs = seqs

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        true_i = torch.from_numpy(np.array([self.pair_data[user_id]]))
        seq = self.seqs[user_id]
        # random,
        weights_pad = np.zeros(np.abs(self.offset))
        if self.offset > 0:  # for b domain,
            rand_weights = np.concatenate((weights_pad, np.ones(self.weights.shape[0])))
            freq_weights = torch.cat([torch.from_numpy(weights_pad), copy.deepcopy(self.weights)])
        else:  # for a domain
            rand_weights = np.concatenate((np.ones(self.weights.shape[0]), weights_pad))
            freq_weights = torch.cat([copy.deepcopy(self.weights), torch.from_numpy(weights_pad)])

        rand_weights[0] = 0
        freq_weights[0] = 0
        for val in seq:
            rand_weights[val] = 0
            freq_weights[val] = 0

        rand_neg = torch.multinomial(torch.from_numpy(rand_weights),
                                     self.num_ng,
                                     replacement=True)
        freq_neg = torch.multinomial(freq_weights,
                                     self.num_ng,
                                     replacement=True)

        return user_id, torch.cat([true_i, rand_neg]), torch.cat([true_i, freq_neg])


class BPRData_EM_se(data.Dataset):
    def __init__(self, features, seqs, num_item, num_ng=0, wf=None):
        super(BPRData_EM_se, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None
        self.seq = seqs
        self.features = features
        self.num_item = num_item
        self.num_ng = num_ng

    def asign_train_mat(self, train_mat):
        self.train_mat = train_mat

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        user = self.features[idx][0]
        true_i = torch.from_numpy(np.array([self.features[idx][1]]))
        seq = self.seq[user]
        # random,
        rand_weights = np.ones(self.weights.shape[0])
        rand_weights[0] = 0
        for val in seq:
            rand_weights[val] = 0
        # freq,
        freq_weights = copy.deepcopy(self.weights)
        freq_weights[0] = 0
        for val in seq:
            freq_weights[val] = 0
        rand_neg = torch.multinomial(torch.from_numpy(rand_weights),
                                     self.num_ng,
                                     replacement=True)
        freq_neg = torch.multinomial(freq_weights,
                                     self.num_ng,
                                     replacement=True)

        return user, torch.cat([true_i, rand_neg]), torch.cat([true_i, freq_neg])


# ############################ Embedding and mapping data loaders
# for training and valid the mapping function
def overlap_user_split(n_over, rate):
    n_val = int(n_over * rate)
    n_train = n_over - n_val
    all_users = np.arange(n_over)
    np.random.shuffle(all_users)
    train_u = all_users[0: n_train]
    val_u = all_users[n_train:]
    return train_u, val_u


class BPRData_EM_over_users(data.Dataset):
    def __init__(self, users):
        super(BPRData_EM_over_users, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
            files = [overlapped, domain-only]
        """
        self.users = users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx]


class BPRData_EM_gan(data.Dataset):
    def __init__(self, num_user_t, num_user_s, overlap_n):
        super(BPRData_EM_gan, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
            files = [overlapped, domain-only]
        """
        self.users_t = np.arange(num_user_t)
        self.users_s = np.arange(num_user_s)
        self.overlap_user = self.users_t[0:overlap_n]
        self.non_overlap_t = self.users_t[overlap_n:]
        self.non_overlap_s = self.users_s[overlap_n:]

    def __len__(self):
        return len(self.overlap_user)

    def __getitem__(self, idx):
        user_o = self.overlap_user[idx]
        user_t = np.random.choice(self.non_overlap_t, 1, replace=False)
        user_s = np.random.choice(self.non_overlap_s, 1, replace=False)
        return user_o, user_t[0], user_s[0]


# ###################################### EMCD cross-domain recommendation after merging user representation.
class BPRData_EM_ct(data.Dataset):
    def __init__(self, seqs, num_item, users, num_ng=0):
        super(BPRData_EM_ct, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
            files = [overlapped, domain-only]
        """
        self.features = []
        for user in users:
            for item in seqs[user]:
                self.features.append([user, item])
        self.num_item = num_item
        self.num_ng = num_ng

    def train_mat_gen(self, user_num, item_num):
        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in self.features:
            train_mat[x[0], x[1]] = 1.0
        self.train_mat = train_mat
        return self.train_mat

    def asign_train_mat(self, train_mat):
        self.train_mat = train_mat

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        user = self.features[idx][0]
        item_i = self.features[idx][1]
        item_j = []
        for idx in range(self.num_ng):
            j = np.random.randint(self.num_item)
            while (user, j) in self.train_mat:
                j = np.random.randint(self.num_item)
            item_j.append(j)

        return user, np.array([item_i]), np.array(item_j)


class BPRData_EM_ce(data.Dataset):
    def __init__(self, features, seqs, num_item, users, num_ng=0, wf=None):
        super(BPRData_EM_ce, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None
        self.seq = seqs
        self.features = []
        for val in features:
            if val[0] in users:
                self.features.append(val)
        self.num_item = num_item
        self.num_ng = num_ng

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        user = self.features[idx][0]
        true_i = torch.from_numpy(np.array([self.features[idx][1]]))
        seq = self.seq[user]
        # random,
        rand_weights = np.ones(self.weights.shape[0])
        rand_weights[0] = 0
        for val in seq:
            rand_weights[val] = 0
        # freq,
        freq_weights = copy.deepcopy(self.weights)
        freq_weights[0] = 0
        for val in seq:
            freq_weights[val] = 0
        rand_neg = torch.multinomial(torch.from_numpy(rand_weights),
                                     self.num_ng,
                                     replacement=True)
        freq_neg = torch.multinomial(freq_weights,
                                     self.num_ng,
                                     replacement=True)

        return user, torch.cat([true_i, rand_neg]), torch.cat([true_i, freq_neg])
