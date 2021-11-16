# -*- coding: UTF-8 -*-
import os


class get_param(object):
    """
    :param args:
                date
                d_model
                n_head
                d_ff
                n_negs
                decoder_neg
                fixed_enc
                lr
                batch_size
                batch_size_val
                result_path
    :return:
    """
    def __init__(self, args):
        # date = "sas_org"
        self.date = args.date
        self.pad_index = 0
        self.num_train_neg = 5  # BPR loss, number of negative samples: k=5
        self.n_bpr_neg = 5
        self.enc_maxlen = 100   # max input seq length to encoder module,
        self.rec_maxlen = 100   # max seq for recommendation module,

        # tf old param
        self.d_ff = 512  # feed-forward dimension
        self.d_model = args.d_model
        self.num_heads = args.n_head
        self.rs_hidden_units = 128
        self.d_k = 32
        self.d_v = 32

        self.dataset_pick = args.dataset_pick     # 1: book_movie, 2: sport_cloth, 3: wesee_txvideo
        num_run = args.run                        # number of run for the same experiments
        self.target_domain = args.target_domain

        domain_pick = "%d%s" % (self.dataset_pick, self.target_domain)

        blocks = {
            "1a": 3, "1b": 3,
            "2a": 3, "2b": 3,
            "3a": 6, "3b": 6,
            "4a": 6, "4b": 6,
            "5a": 6, "5b": 6,
            "6a": 6, "6b": 6,
        }

        self.num_blocks = blocks[domain_pick]  # number of Transformer layer for enc and dec
        dataset_all = {
            1: "book_movie",
            2: "sport_cloth",
            3: "business_10",
            4: "business_30",
            5: "business_50",
            6: "business_75",
        }

        domains = {"1a": "movie", "1b": "book",
                   "2a": "sport", "2b": "cloth",
                   "3a": "wesee", "3b": "txvideo",
                   "4a": "wesee", "4b": "txvideo",
                   "5a": "wesee", "5b": "txvideo",
                   "6a": "wesee", "6b": "txvideo"
                   }

        shared_dec = True
        if shared_dec:  # use the decoder as recommendation model.
            self.rs_d_model = args.d_model
            self.rs_num_blocks = self.num_blocks
            self.rs_d_ff = args.d_ff
        else:  # if we use different model for rec and dec, then reconfigure hyper-parameters for rec model
            self.rs_d_model = 256
            self.rs_num_blocks = 3
            self.rs_d_ff = 256

        # n_negs = 30         # when optimize reconstruction loss, we use sampled softmax loss function.
        self.n_negs = args.n_negs

        self.dataset = dataset_all[self.dataset_pick]
        self.domain_name = domains[domain_pick]

        # self.decoder_neg = True  # whether to use negative sampling while training enc-dec
        # self.fixed_enc = True    # whether the shared_enc can be updated by rs loss only
        self.decoder_neg = args.decoder_neg  # whether to use negative sampling while training enc-dec
        self.fixed_enc = args.fix_enc      # whether the shared_enc can be updated by rs loss only

        # training setting
        self.lr_rs = args.lr

        self.batch_size = args.batch_size  #
        self.batch_size_val = args.batch_size_val
        # training_steps_tune = 15000  # steps on Rec task
        if args.cross == "True" or "business" in self.dataset:
            self.result_path = os.path.join(args.result_path, "%s_%s_%d_%d_mg" % (self.dataset,
                                                                                  self.domain_name,
                                                                                  args.d_model,
                                                                                  num_run)
                                            )
        else:
            self.result_path = os.path.join(args.result_path, "%s_%d_%d_mg" % (self.domain_name,
                                                                               args.d_model, num_run))

        if not os.path.isdir(self.result_path):
            os.mkdir(self.result_path)
        # Data set configuration public
        if "business" in self.dataset:
            rate = self.dataset.split("_")[1]
            self.data_path = "/data/ceph/seqrec/data/business/kdd_%s_f" % rate
            data_info = {
                "10": [194756, 1203194, 1434999, 288155, 1311440],
                "30": [585580, 1398645, 1630372, 303942, 1356831],
                "50": [976556, 1594256, 1825737, 316338, 1397521],
                "75": [1464731, 1838276, 2069892, 329679, 1438803]
            }
            self.num_overlap_users = data_info[rate][0]
            self.num_users_a = data_info[rate][1]
            self.num_users_b = data_info[rate][2]
            self.vocab_size_a = data_info[rate][3] + 1   # a domain (wesee)
            self.vocab_size_b = data_info[rate][4] + 1   # b domain (txvideo)
        else:
            if self.dataset == "sport_cloth":
                self.data_path = "/data/ceph/seqrec/data_guru/public/Amazon_torch/sport_cloth"
                self.vocab_size_a = 11835 + 1
                self.num_users_a = 9024
                self.vocab_size_b = 42139 + 1
                self.num_users_b = 46810       # inter
                self.num_overlap_users = 1062
            else:
                self.data_path = "/data/ceph/seqrec/data_guru/public/Amazon_torch/movie_book"
                self.vocab_size_a = 5536 + 1
                self.num_users_a = 4261
                self.vocab_size_b = 51366 + 1  # Book
                self.num_users_b = 42940
                self.num_overlap_users = 584

        if self.target_domain == "a":
            self.vocab_size = self.vocab_size_a
            self.num_users = self.num_users_a
        else:
            self.vocab_size = self.vocab_size_b
            self.num_users = self.num_users_b
        # If this is for cross domain
        self.domain_name_a = domains[str(self.dataset_pick) + "a"]
        self.domain_name_b = domains[str(self.dataset_pick) + "b"]

        self.logdir = os.path.join(self.result_path, self.dataset)

        self.model_path = os.path.join(self.result_path, "model")
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        if "business" in self.dataset:
            if args.sas == "True":
                self.training_steps = 3000
                self.training_steps_tune = 3000
                self.batch_size_over = 100
            elif args.cross != "True":
                self.training_steps = 2000
                self.training_steps_tune = 2000
                self.batch_size_over = 100
                self.n_warmup_steps = 1000
            else:
                self.training_steps = 2000
                self.training_steps_tune = 3000
                self.batch_size_over = 100
                self.n_warmup_steps = 1000
                # testing
                rate = float(self.dataset.split("_")[1]) / 100
                print(rate)
                self.batch_size_over = int(self.batch_size * rate)
                if self.batch_size_over < 10:
                    self.batch_size_over = 10
        else:
            if args.sas == "True":
                # self.training_steps does not matter in sas model.
                self.training_steps = 200 * int(self.num_users / self.batch_size) + 1       # For reconstruction training
                self.training_steps_tune = 600 * int(self.num_users / self.batch_size) + 1  # For recommendation training
                self.batch_size_over = 100
            elif args.cross != "True":
                self.training_steps = 500 * int(self.num_users / self.batch_size) + 1
                self.training_steps_tune = 500 * int(self.num_users / self.batch_size) + 1
                self.n_warmup_steps = 1000
            else:
                users_n = self.num_users_a + self.num_users_b
                self.training_steps = 300 * int(users_n / self.batch_size) + 1
                self.training_steps_tune = 400 * int(users_n / self.batch_size) + 1
                # testing
                # self.training_steps = 200
                # self.training_steps_tune = 200

                self.eval_step = int(users_n / (self.batch_size * 5))   # (averaged 10 epochs.)
                self.n_warmup_steps = int(self.training_steps / 2)
                ratio = int(users_n / self.num_overlap_users)
                self.batch_size_over = int(self.batch_size / ratio) + 1
                if self.batch_size_over < 10:
                    self.batch_size_over = 10
                # test parameters:
        if self.domain_name == "movie":
            self.freq_train_ep = 400
        elif self.domain_name == "cloth":
            self.freq_train_ep = 200
            self.training_steps_tune += 100
        elif self.domain_name == "book":
            self.freq_train_ep = 200
            self.training_steps_tune += 100
        else:       # sport
            self.freq_train_ep = 350
        # relations between batch_size and # of epoch
        # 256 100
        # 512 200
        # 1024 400
        if "business" in self.dataset:
            self.eval_epoch = 70   # steps
            self.eval_steps = int(50000 / self.batch_size_val) + 1  # evaluate on 2w user each time.
        else:
            self.eval_epoch = 10
            self.eval_steps = int(2000 / self.batch_size_val) + 1   # evaluate on 2k user each time.

        # This is warming up setting for reconstruction training.
        self.dropout_rate = 0.5
        self.smoothing = 0.1

        # number of negative samples in testing.
        if "business" in self.dataset:
            self.candidate_size = 19999
        else:
            self.candidate_size = 199
        # Discriminator parameters:
        self.dis_dim = self.d_model * 5
        # self.n_gpus = args.n_gpu

        self.training_steps_tune = 300  # for testing
