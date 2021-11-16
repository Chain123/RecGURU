# -*- coding: UTF-8 -*-
"""
AutoEnc4Rec model for sequential recommendation:
Encoder -> embedding -> sequence decoder.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

import AutoEnc4Rec_cross as Model
import config_auto4rec as param_c
import data.data_loader as Dataloader
import gan_training as gt
import tools.lossfunctions as lf
import tools.utils as ut
import Transformer.transformer as all_module


def str2bool_par(val):
    if val == "True":
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--date", type=str, default="sas_org", help="labeling")
    parser.add_argument("--d_model", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--n_head", type=int, default=1, help="multi-headed number of head")
    parser.add_argument("--d_ff", type=int, default=512, help="feed-forward dim")
    parser.add_argument("--n_negs", type=int, default=30, help="number of negs in sampled softmax")
    parser.add_argument("--decoder_neg", type=bool, default=True,
                        help="whether to use negative sampling while training enc-dec")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=256, help="batch size for training")
    parser.add_argument("--target_domain", type=str, default="a", help="domain")
    parser.add_argument("--dataset_pick", type=int, default=1,
                        help="1 for movie_book, 2 for sport_cloth, 2 for wesee-txvideo")
    parser.add_argument("--run", type=int, default=1, help="number of runs")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus")
    parser.add_argument("--result_path", type=str, default="/data/ceph/seqrec/torch/result/gur_s/non_shared/",
                        help="result folder")
    parser.add_argument("--sas", type=str, default="False", help="whether to use sas")
    parser.add_argument("--cross", type=str, default="False", help="whether to use sas")
    parser.add_argument("--enc_share", type=str, default="True", help="whether to use sas")
    parser.add_argument("--share_dec", type=str, default="False", help="whether to use sas")
    parser.add_argument("--fix_enc", type=str, default="True", help="whether to use sas")
    args = parser.parse_args()
    args.fix_enc = str2bool_par(args.fix_enc)
    param = param_c.get_param(args)
    loss_ce = nn.CrossEntropyLoss(reduction="none")
    loss_s_ce = lf.SampledCrossEntropyLoss(reduction="none")
    loss_bpr = lf.BPRLoss()

    # config data and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    all_files_a = [os.path.join(param.data_path, name) for name in os.listdir(param.data_path) if
                   param.domain_name_a in name and "_" not in name]
    freq_file_a = [os.path.join(param.data_path, name) for name in os.listdir(param.data_path) if
                   param.domain_name_a in name and "freq" in name]
    item_fre_a = Dataloader.load_pickle(freq_file_a[0])
    # item_fre_a= None

    all_files_b = [os.path.join(param.data_path, name) for name in os.listdir(param.data_path) if
                   param.domain_name_b in name and "_" not in name]
    freq_file_b = [os.path.join(param.data_path, name) for name in os.listdir(param.data_path) if
                   param.domain_name_b in name and "freq" in name]
    item_fre_b = Dataloader.load_pickle(freq_file_b[0])
    # item_fre_b = None
    overlap_file = [os.path.join(param.data_path, name) for name in os.listdir(param.data_path) if
                    param.domain_name_a in name and param.domain_name_b in name]
    print("=================")
    print(all_files_a, freq_file_a)
    print(all_files_b, freq_file_b)
    print(overlap_file)
    print("*****************")
    train_loader_AE_a = Dataloader.dataloader_gen(all_files_a, param, param.n_negs, train=True,
                                                  seq_len=None, wf=None, domain="a", n_w=8)
    train_loader_RE_a = Dataloader.dataloader_gen(all_files_a, param, param.n_bpr_neg, train=True,
                                                  seq_len=None, wf=None, rec=True, domain="a", n_w=8)
    train_loader_re_f_a = Dataloader.dataloader_gen(all_files_a, param, param.n_bpr_neg, train=True,
                                                    seq_len=None, wf=item_fre_a, rec=True, domain="a", n_w=8)
    train_loader_re_test_a = Dataloader.dataloader_gen(all_files_a, param, param.n_bpr_neg, train=False,
                                                       seq_len=None, wf=item_fre_a, domain="a", n_w=8)
    # b domain data
    train_loader_AE_b = Dataloader.dataloader_gen(all_files_b, param, param.n_negs,
                                                  train=True, seq_len=None, wf=None, domain="b", n_w=8)
    train_loader_RE_b = Dataloader.dataloader_gen(all_files_b, param, param.n_bpr_neg,
                                                  train=True, seq_len=None, wf=None, rec=True, domain="b", n_w=8)
    train_loader_re_f_b = Dataloader.dataloader_gen(all_files_b, param, param.n_bpr_neg, train=True,
                                                    seq_len=None, wf=item_fre_b, rec=True, domain="b", n_w=8)
    train_loader_re_test_b = Dataloader.dataloader_gen(all_files_b, param, param.n_bpr_neg,
                                                       train=False, seq_len=None, wf=item_fre_b, domain="b", n_w=8)
    # overlap data.
    train_overlap = Dataloader.dataloader_gen_over(overlap_file, param, n_w=1)

    ae_loaders = [train_loader_AE_a, train_loader_AE_b]
    if args.target_domain == "a":
        rec_loaders = [train_loader_RE_a, train_loader_re_f_a]
        test_loaders = train_loader_re_test_a
    else:
        rec_loaders = [train_loader_RE_b, train_loader_re_f_b]
        test_loaders = train_loader_re_test_b
    # generator
    # devices = list(range(args.n_gpu))
    # print(devices)
    if args.enc_share == "False":
        enc_share = False
    else:
        enc_share = True
    enc_model = Model.MyAuto4Rec_c(device,
                                   param,
                                   wf=None,
                                   enc_share=enc_share,
                                   dec_rec=False).to(torch.float32).to(device)
    if torch.cuda.device_count() > 1:
        enc_model = nn.DataParallel(enc_model)
    opt_rec = all_module.ScheduledOptim(optim.Adam(enc_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                                        1.0, param.d_model, param.n_warmup_steps)
    opt_gen = optim.Adam(enc_model.parameters(), lr=0.0001, betas=(0.5, 0.9))

    # dis
    netD = ut.Discriminator(param.d_model, 1, param.dis_dim).to(torch.float32).to(device)
    if torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)
    opt_dis = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.9))
    # only adversarial training is applied.
    # gt.train_gan(enc_model, netD, train_loader_a, train_loader_b, opt_enc, optimizerD, device, param.batch_size, 5000)

    # gt.main(enc_model, opt_rec, netD, opt_gen, opt_dis, param, device, loaders)
    gt.main_2(enc_model, opt_rec, netD, opt_gen, opt_dis, param, device,
              ae_loaders, rec_loaders, test_loaders, train_overlap)
    # gt.main_3(enc_model, opt_rec, netD, opt_gen, opt_dis, param, device,
    #           ae_loaders, rec_loaders, test_loaders, train_overlap, devices)
