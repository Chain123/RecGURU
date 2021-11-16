# -*- coding: UTF-8 -*-
"""
AutoEnc4Rec model for sequential recommendation:
Encoder -> embedding -> sequence decoder.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor as FT
# import config_auto4rec as param
from tqdm import tqdm

import Transformer.transformer as all_module

loss_ce = nn.CrossEntropyLoss(reduction="none")


class MyRec(nn.Module):

    def __init__(self, device_t, param, wf=None, dec_rec=False,
                 fix_enc=False, sas=False, pos_train=False):
        """
        """
        super(MyRec, self).__init__()
        self.fix_enc = fix_enc
        self.sas = sas
        self.device = device_t
        if sas:
            self.src_emb = nn.Embedding(param.vocab_size + 1, param.d_model, padding_idx=param.pad_index)
            if pos_train:
                self.pos_emb = all_module.PositionalEncodingM(d_model=param.d_model, dropout=param.dropout_rate,
                                                              max_len=param.enc_maxlen, device=device_t)
            else:
                self.pos_emb = all_module.PositionalEncoding(d_model=param.d_model, dropout=param.dropout_rate)
            self.sas_rec = all_module.EncoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device_t, dropout=param.dropout_rate)
        else:
            self.AutoEnc = MyAuto4Rec(vocab_size=param.vocab_size, d_model=param.d_model, pad_index=0, d_ff=param.d_ff,
                                      d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads, n_layers=param.num_blocks,
                                      device=torch.device(device_t), param=param, wf=wf, pos_train=pos_train)
            if not dec_rec:  # use decoder as recommendation model.
                self.recommend = all_module.DecoderM(
                    d_model=param.d_model, d_ff=param.d_ff,
                    d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                    n_layers=param.num_blocks, pad_index=param.pad_index,
                    device=device_t, dropout=param.dropout_rate)
        self.dec_rec = dec_rec
        self.param = param

    def get_embedding(self, enc_in, dec_in):
        """
        :param enc_in:  the input seq to enc
        :param dec_in:  the input seq of recommender model
        :return::       the embedding of input seq to recommender model
        func: use for recommendation.
        """
        if self.dec_rec:
            # share dec as rec
            dec_outputs, _, _ = self.AutoEnc.get_dec_out(enc_in, dec_in)
        else:
            enc_outputs, _ = self.AutoEnc.get_seq_embed(enc_in)
            enc_outputs = torch.unsqueeze(enc_outputs[:, -1, :], 1).repeat(1, self.param.enc_maxlen, 1)
            # detach， to fix the encoder unchanged when training recommendation model.
            if self.fix_enc:
                enc_outputs = enc_outputs.detach()
            mask = dec_in == self.param.pad_index
            mask = (1 - mask.to(int)).to(torch.float32)  # 0, 1
            mask.to(self.device)
            dec_outputs = self.AutoEnc.src_emb(dec_in)
            dec_outputs = self.AutoEnc.pos_emb(dec_outputs, mask)
            #
            dec_self_attn_pad_mask = all_module.get_attn_pad_mask(dec_in, dec_in, self.param.pad_index)
            dec_self_attn_subsequent_mask = all_module.get_attn_subsequent_mask(dec_in)
            dec_self_attn_subsequent_mask = dec_self_attn_subsequent_mask.to(self.device)
            dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
            dec_enc_attn_mask = all_module.get_attn_pad_mask(dec_in, enc_in, self.param.pad_index)
            dec_outputs, _, _ = self.recommend(dec_outputs, enc_in, enc_outputs,
                                               dec_self_attn_mask, dec_enc_attn_mask, mask)

        return dec_outputs

    def get_embedding_sas(self, dec_in):
        dec_mask = all_module.get_attn_pad_mask(dec_in, dec_in, self.param.pad_index)
        pad_mask = dec_in == self.param.pad_index
        pad_mask.to(self.device)
        pad_mask = (1 - pad_mask.to(int)).to(torch.float32)  # 0, 1
        dec_embedding = self.src_emb(dec_in)
        dec_embedding = self.pos_emb(dec_embedding, pad_mask)

        dec_embedding, _, = self.sas_rec(dec_embedding, dec_mask, pad_mask)
        return dec_embedding

    def forward(self, enc_in, dec_in, dec_out, n_items, recon=False):
        """
        :param recon:   Is this for recommendation or reconstruction.
        :param n_items: negative items
        :param enc_in:  the embedding of enc_seq
        :param dec_in:  the input seq of decoder model
        :param dec_out: the output/target seq of decoder model
        :return:        the recommendation loss (or logits that can compute this loss)

        reconmendation forward:  so "num_train_neg" is used.
        """
        if recon:
            return self.AutoEnc(enc_in, dec_in, dec_out, n_items)
            #     enc_inputs, dec_inputs, dec_outputs, n_items
        else:
            if self.sas:
                dec_in = self.get_embedding_sas(dec_in)  # [B, len, d_model]
                n_items = self.src_emb(n_items).view(-1, self.param.enc_maxlen,
                                                     self.param.num_train_neg,
                                                     self.param.d_model)
                dec_out = self.src_emb(dec_out).view(-1, self.param.enc_maxlen,
                                                     1, self.param.d_model)
                # p_embeddings = torch.unsqueeze(p_embeddings, 2)
            else:
                dec_in = self.get_embedding(enc_in, dec_in)  # [B, len, d_model]
                n_items = self.AutoEnc.src_emb(n_items).view(-1, self.param.enc_maxlen,
                                                             self.param.num_train_neg,
                                                             self.param.d_model)
                dec_out = self.AutoEnc.src_emb(dec_out).view(-1, self.param.enc_maxlen,
                                                             1, self.param.d_model)

            dec_in = dec_in.view(-1, self.param.enc_maxlen, 1, self.param.d_model)
            p_logits = torch.squeeze(torch.matmul(dec_in, dec_out.transpose(2, 3)), 2)  # [b, len, 1, 1]
            n_logits = torch.squeeze(torch.matmul(dec_in, n_items.transpose(2, 3)), 2)  # [b, len, 1, neg]
            # logits = torch.cat([p_logits, n_logits], dim=2)  # [b, len, neg + 1]
            return p_logits, n_logits


class MyAuto4Rec(nn.Module):

    def __init__(self, vocab_size, d_model, pad_index, d_ff, d_k, d_v, n_heads,
                 n_layers, device, param, wf=None, pos_train=False):
        """
        """
        super(MyAuto4Rec, self).__init__()
        self.pad_index = pad_index
        self.param = param
        self.device = device
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None

        self.src_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_index)
        # print("# embedding table,", self.src_emb.weight.shape)
        if pos_train:
            self.pos_emb = all_module.PositionalEncodingM(d_model=d_model, dropout=param.dropout_rate,
                                                          max_len=param.enc_maxlen, device=device)
        else:
            self.pos_emb = all_module.PositionalEncoding(d_model=d_model, dropout=param.dropout_rate)

        self.encoder = all_module.EncoderM(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=pad_index,
            device=device, dropout=param.dropout_rate)

        self.decoder = all_module.DecoderM(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=pad_index,
            device=device, dropout=param.dropout_rate)
        # if not self.param.decoder_neg:
        #     self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def get_seq_embed(self, enc_inputs):
        mask = enc_inputs == self.pad_index
        mask = (1 - mask.to(int)).to(torch.float32)  # 0, 1
        mask.to(self.device)
        enc_self_attn_mask = all_module.get_attn_pad_mask(enc_inputs, enc_inputs, self.pad_index)
        enc_inputs = self.src_emb(enc_inputs)
        enc_inputs = self.pos_emb(enc_inputs, mask)

        enc_inputs, enc_self_attns = self.encoder(enc_inputs, enc_self_attn_mask, mask)
        return enc_inputs, enc_self_attns

    def get_dec_out(self, enc_inputs, dec_inputs):
        enc_outputs, _ = self.get_seq_embed(enc_inputs)
        enc_outputs = torch.unsqueeze(enc_outputs[:, -1, :], 1).repeat(1, self.param.enc_maxlen, 1)
        mask = dec_inputs == self.pad_index
        mask = (1 - mask.to(int)).to(torch.float32)  # 0, 1
        mask.to(self.device)
        dec_outputs = self.src_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs, mask)
        #
        dec_self_attn_pad_mask = all_module.get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = all_module.get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_subsequent_mask = dec_self_attn_subsequent_mask.to(self.device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = all_module.get_attn_pad_mask(dec_inputs, enc_inputs, self.pad_index)

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_outputs, enc_inputs, enc_outputs,
                                                                  dec_self_attn_mask, dec_enc_attn_mask, mask)
        # dec_logits = self.projection(dec_outputs)  # last
        return dec_outputs, dec_self_attns, dec_enc_attns

    def forward(self, enc_inputs, dec_inputs, dec_outputs, n_items):
        """
        :param n_items:     negative items
        :param dec_outputs:
        :param enc_inputs: input seq to encoder
        :param dec_inputs: input seq to decoder
        :return: reconstruction logits at each position.

        reconstruction forward:
        """
        dec_inputs, _, _ = self.get_dec_out(enc_inputs, dec_inputs)  # [B, len, d_model]
        # TODO form loss.
        if self.param.decoder_neg and self.param.n_negs < self.param.vocab_size:
            n_items = self.src_emb(n_items).view(-1, self.param.enc_maxlen,
                                                 self.param.n_negs, self.param.d_model)
            dec_outputs = self.src_emb(dec_outputs).view(-1, self.param.enc_maxlen,
                                                         1, self.param.d_model)

            dec_inputs = dec_inputs.view(-1, self.param.enc_maxlen, 1, self.param.d_model)
            p_logits = torch.squeeze(torch.matmul(dec_inputs, dec_outputs.transpose(2, 3)), 2)  # [b, len, 1, 1]
            n_logits = torch.squeeze(torch.matmul(dec_inputs, n_items.transpose(2, 3)), 2)  # [b, len, 1, neg]
            return torch.cat([p_logits, n_logits], dim=2)  # [b, len, neg + 1]
        else:
            print("all vocab logits")
            return torch.matmul(dec_inputs, self.src_emb.weight.transpose(1, 0))  # [b, len, d_mode] [n_vocab, d_model]


def train_ae(model_train, opt, steps, data, param, device, neg_sample=True):
    """
    :param device:
    :param param:
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
        n_items, val, test = n_items.to(device), val.to(device), test.to(device)
        opt.zero_grad()
        enc_in, dec_in, dec_out = seqs[0], seqs[1], seqs[2]
        bs, sl = dec_out.shape[0], dec_out.shape[1]
        enc_in, dec_in, dec_out = enc_in.to(device), dec_in.to(device), dec_out.to(device)
        logits = model_train(enc_in, dec_in, dec_out, n_items)
        if neg_sample:
            # negative sample use fake label
            label = torch.zeros(bs * sl).long().to(device)
            num_class = param.n_negs + 1
        else:
            # softmax on whole vocab, use true lable
            label = dec_out.view(-1, 1).long()
            num_class = param.vocab_size
        mask = dec_out == param.pad_index
        mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
        mask.to(device)
        # cross-entropy loss
        loss = loss_ce(logits.view(-1, num_class), label)
        loss_m = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
        # back propagation
        loss_m.backward()
        # update
        # opt.step()  # normal optimizer (Adam)
        opt.step_and_update_lr()
        if device == "cpu":
            loss_epoch = loss_m.detach().numpy()
        else:
            loss_epoch = loss_m.cpu().detach().numpy()
        # eval step.
        if i % 50 == 49:
            print("reconstruction loss after %d batch" % i, loss_epoch)


def main(param, device_t, train_loader_t, item_freq=None):
    model = MyAuto4Rec(vocab_size=param.vocab_size, d_model=param.d_model, pad_index=0, d_ff=param.d_ff,
                       d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads, n_layers=param.num_blocks,
                       device=torch.device(device_t), param=param, wf=item_freq, pos_train=False)
    model = model.to(torch.float32)
    model.to(device_t)
    # word frequency.
    # opt = optim.Adam(model.parameters(), lr=param.lr, weight_decay=1e-4)  # normal optimizer.
    opt = all_module.ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        1.0, param.d_model, param.n_warmup_steps)
    train_ae(model, opt, param.training_steps, train_loader_t, param, device_t, neg_sample=True)

