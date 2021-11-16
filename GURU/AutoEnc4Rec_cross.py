# -*- coding: UTF-8 -*-
"""
AutoEnc4Rec model for sequential recommendation:
Encoder -> embedding -> sequence decoder.
"""

import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor as FT

import Transformer.transformer as all_module

loss_ce = nn.CrossEntropyLoss(reduction="none")


class MyAuto4Rec_c(nn.Module):

    def __init__(self, device, param, wf=None, dec_share=False, dec_rec=False, enc_share=True):
        super(MyAuto4Rec_c, self).__init__()
        self.param = param
        self.device = device
        self.dec_share = dec_share
        self.dec_rec = dec_rec
        if wf is not None:
            wf = np.power(wf, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)
        else:
            self.weights = None

        self.src_emb_a = nn.Embedding(param.vocab_size_a + 1, param.d_model)
        self.pos_emb_a = all_module.PositionalEncoding(d_model=param.d_model, dropout=param.dropout_rate)

        self.src_emb_b = nn.Embedding(param.vocab_size_b + 1, param.d_model)
        self.pos_emb_b = all_module.PositionalEncoding(d_model=param.d_model, dropout=param.dropout_rate)
        # shared encoder model
        self.enc_share = enc_share
        if enc_share:
            self.encoder = all_module.EncoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)
        else:
            self.encoder_a = all_module.EncoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)
            self.encoder_b = all_module.EncoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)
        # shared or separated decoder model
        if dec_share:
            self.decoder = all_module.DecoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)
        else:
            self.decoder_a = all_module.DecoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)

            self.decoder_b = all_module.DecoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)
        # output layer
        if not self.param.decoder_neg:
            self.projection_a = nn.Linear(param.d_model, param.vocab_size_a, bias=False)
            self.projection_b = nn.Linear(param.d_model, param.vocab_size_b, bias=False)

        # recommendation model
        if not dec_rec:
            self.recommend_a = all_module.DecoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)
            self.recommend_b = all_module.DecoderM(
                d_model=param.d_model, d_ff=param.d_ff,
                d_k=param.d_k, d_v=param.d_v, n_heads=param.num_heads,
                n_layers=param.num_blocks, pad_index=param.pad_index,
                device=device, dropout=param.dropout_rate)

    def get_seq_embed(self, enc_inputs, domain="a", mask=None):
        # mask = enc_inputs == self.param.pad_index
        # mask = (1 - mask.to(int)).to(torch.float32)  # 0, 1
        # mask.to(self.device)
        if domain == "a":
            enc_outputs = self.src_emb_a(enc_inputs)
            enc_outputs = self.pos_emb_a(enc_outputs, mask)
        else:
            enc_outputs = self.src_emb_b(enc_inputs)
            enc_outputs = self.pos_emb_b(enc_outputs, mask)
        if domain == "a":
            pad_index = self.param.vocab_size_a
        else:
            pad_index = self.param.vocab_size_b
        enc_self_attn_mask = all_module.get_attn_pad_mask(enc_inputs, enc_inputs, pad_index)
        if self.enc_share:
            enc_outputs, enc_self_attns = self.encoder(enc_outputs, enc_self_attn_mask, pad_mask=mask)
        else:
            if domain == "a":
                enc_outputs, enc_self_attns = self.encoder_a(enc_outputs, enc_self_attn_mask, pad_mask=mask)
            else:
                enc_outputs, enc_self_attns = self.encoder_b(enc_outputs, enc_self_attn_mask, pad_mask=mask)
        return enc_outputs

    def get_dec_out(self, enc_inputs, dec_inputs, domain="a", mask=None):
        d_mask = enc_inputs == self.param.pad_index
        d_mask = (1 - d_mask.to(int)).to(torch.float32)  # 0, 1
        d_mask.to(self.device)
        enc_outputs = self.get_seq_embed(enc_inputs, domain, mask.view(-1, self.param.enc_maxlen))
        enc_outputs = torch.unsqueeze(enc_outputs[:, -1, :], 1).repeat(1, self.param.enc_maxlen, 1)
        if domain == "a":
            dec_outputs = self.src_emb_a(dec_inputs)
            dec_outputs = self.pos_emb_a(dec_outputs, d_mask.view(-1, self.param.enc_maxlen))
        else:
            dec_outputs = self.src_emb_b(dec_inputs)
            dec_outputs = self.pos_emb_b(dec_outputs, d_mask.view(-1, self.param.enc_maxlen))
        #
        dec_self_attn_pad_mask = all_module.get_attn_pad_mask(dec_inputs, dec_inputs, self.param.pad_index)
        dec_self_attn_subsequent_mask = all_module.get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_subsequent_mask = dec_self_attn_subsequent_mask.to(self.device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = all_module.get_attn_pad_mask(dec_inputs, enc_inputs, self.param.pad_index)
        if self.dec_share:
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_outputs, enc_inputs, enc_outputs,
                                                                      dec_self_attn_mask, dec_enc_attn_mask)
        elif domain == "a":
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder_a(dec_outputs, enc_inputs, enc_outputs,
                                                                        dec_self_attn_mask, dec_enc_attn_mask,
                                                                        pad_m=d_mask)
        else:
            dec_outputs, dec_self_attns, dec_enc_attns = self.decoder_b(dec_outputs, enc_inputs, enc_outputs,
                                                                        dec_self_attn_mask, dec_enc_attn_mask,
                                                                        pad_m=d_mask)
        # dec_logits = self.projection(dec_outputs)  # last
        return dec_outputs, dec_self_attns, dec_enc_attns

    def recommend_forward(self, enc_in, dec_in, domain, mask):
        d_mask = dec_in == self.param.pad_index
        d_mask = (1 - d_mask.to(int)).to(torch.float32)  # 0, 1
        d_mask.to(self.device)
        enc_outputs = self.get_seq_embed(enc_in, domain, mask.view(-1, self.param.enc_maxlen))
        enc_outputs = torch.unsqueeze(enc_outputs[:, -1, :], 1).repeat(1, self.param.enc_maxlen, 1)
        if domain == "a":
            dec_outputs = self.src_emb_a(dec_in)
            dec_outputs = self.pos_emb_a(dec_outputs, d_mask.view(-1, self.param.enc_maxlen))
        else:
            dec_outputs = self.src_emb_b(dec_in)
            dec_outputs = self.pos_emb_b(dec_outputs, d_mask.view(-1, self.param.enc_maxlen))
        #
        if self.param.fixed_enc:
            enc_outputs = enc_outputs.detach()
        dec_self_attn_pad_mask = all_module.get_attn_pad_mask(dec_in, dec_in, self.param.pad_index)
        dec_self_attn_subsequent_mask = all_module.get_attn_subsequent_mask(dec_in)
        dec_self_attn_subsequent_mask = dec_self_attn_subsequent_mask.to(self.device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = all_module.get_attn_pad_mask(dec_in, enc_in, self.param.pad_index)
        if domain == "a":  # TODO will never share the decoder across domains.
            if self.dec_rec:
                dec_outputs, _, _ = self.decoder_a(dec_outputs, enc_in, enc_outputs, dec_self_attn_mask,
                                                   dec_enc_attn_mask, d_mask)
            else:
                dec_outputs, _, _ = self.recommend_a(dec_outputs, enc_in, enc_outputs, dec_self_attn_mask,
                                                     dec_enc_attn_mask, d_mask)
        else:
            if self.dec_rec:
                dec_outputs, _, _ = self.decoder_b(dec_outputs, enc_in, enc_outputs, dec_self_attn_mask,
                                                   dec_enc_attn_mask, d_mask)
            else:
                dec_outputs, _, _ = self.recommend_b(dec_outputs, enc_in, enc_outputs, dec_self_attn_mask,
                                                     dec_enc_attn_mask, d_mask)
        return dec_outputs

    def forward(self, enc_inputs, dec_inputs, dec_outputs, n_items, domain, mask):
        """
        :param mask:
        :param domain:      forward for which domain.
        :param n_items:     negative items
        :param dec_outputs:
        :param enc_inputs: input seq to encoder
        :param dec_inputs: input seq to decoder
        :return: reconstruction logits at each position.
        """
        dec_out, _, _ = self.get_dec_out(enc_inputs, dec_inputs, domain, mask)  # [B, len, d_model]
        # TODO form loss.
        if domain == "a":
            vocab_size = self.param.vocab_size_a
        else:
            vocab_size = self.param.vocab_size_b
        if self.param.decoder_neg and self.param.n_negs < vocab_size:
            if domain == "a":
                n_embeddings = self.src_emb_a(n_items).view(-1, self.param.enc_maxlen,
                                                            self.param.n_negs, self.param.d_model)
                p_embeddings = self.src_emb_a(dec_outputs).view(-1, self.param.enc_maxlen,
                                                                1, self.param.d_model)
            else:
                n_embeddings = self.src_emb_b(n_items).view(-1, self.param.enc_maxlen,
                                                            self.param.n_negs, self.param.d_model)
                p_embeddings = self.src_emb_b(dec_outputs).view(-1, self.param.enc_maxlen,
                                                                1, self.param.d_model)
            dec_out = dec_out.view(-1, self.param.enc_maxlen, 1, self.param.d_model)
            p_logits = torch.squeeze(torch.matmul(dec_out, p_embeddings.transpose(2, 3)), 2)  # [b, len, 1, 1]
            n_logits = torch.squeeze(torch.matmul(dec_out, n_embeddings.transpose(2, 3)), 2)  # [b, len, 1, neg]
            logits = torch.cat([p_logits, n_logits], dim=2)  # [b, len, neg + 1]
        else:
            if domain == "a":
                logits = self.projection_a(dec_out)  # [b, len, vocab_size]
            else:
                logits = self.projection_b(dec_out)  # [b, len, vocab_size]
        return logits
