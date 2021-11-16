# -*- coding: UTF-8 -*-

"""
Transformer modules: attention is all you need.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.current_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def get_lr(self):
        return self.current_lr

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.current_lr = lr

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


def get_attn_pad_mask(seq_q, seq_k, pad_index):
    """
    :param seq_q: q sequence of multi-headed attention layer: [B, seq_len]
    :param seq_k: k sequence of multi-headed attention layer
    :param pad_index: values of the [pad] token
    :return: pad mask [B, seq_len, seq_len]
    1: masked position
    0: non-masked position
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_index).unsqueeze(1)
    pad_attn_mask = torch.as_tensor(pad_attn_mask, dtype=torch.int)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    """
    :param seq: Future mask
    :return:
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).int()
    return subsequent_mask


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEncoding(nn.Module):
    """Implement the fixed PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, mask):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x * torch.unsqueeze(mask, 2))


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, device, dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.device = device
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
        attn_mask = attn_mask.to(self.device)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # TODO drop out
        if self.dropout_rate is not None:
            attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, dropout_rate=None):
        super(MultiHeadAttention, self).__init__()
        self.WQ = nn.Linear(d_model, d_k * n_heads)
        self.WK = nn.Linear(d_model, d_k * n_heads)
        self.WV = nn.Linear(d_model, d_v * n_heads)

        self.linear = nn.Linear(n_heads * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)
        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(d_k, device, dropout_rate)  # TODO ADD

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.ScaledDotProductAttention(Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)  # TODO: ADD self,
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + Q), attn


class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff, dropout_rate=None):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:  # TODO or use batch normalization.
            self.dropout1 = torch.nn.Dropout(p=dropout_rate)
            self.dropout2 = torch.nn.Dropout(p=dropout_rate)

        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)

        self.relu = GELU()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-8)

    def forward(self, inputs):
        residual = inputs
        output = self.l1(inputs)
        if self.dropout_rate is not None:
            output = self.dropout1(output)
        output = self.relu(output)
        output = self.l2(output)
        if self.dropout_rate is not None:
            output = self.dropout2(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, dropout):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,
            device=device, dropout_rate=dropout)
        self.pos_ffn = PositionWiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            Q=enc_inputs, K=enc_inputs,
            V=enc_inputs, attn_mask=enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device):
        super(Encoder, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model=d_model, dropout=0)

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(d_model=d_model, d_ff=d_ff,
                                         d_k=d_k, d_v=d_v, n_heads=n_heads,
                                         device=device)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        enc_outputs = self.src_emb(x)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(x, x, self.pad_index)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device, dropout):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,
            device=device, dropout_rate=dropout)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,
            device=device, dropout_rate=dropout)
        self.pos_ffn = PositionWiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, dropout_rate=dropout)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device):
        super(Decoder, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.tgt_emb = nn.Embedding(
            vocab_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.pad_index)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns = dec_enc_attns.permute([1, 0, 2, 3, 4])

        return dec_outputs, dec_self_attns, dec_enc_attns


class MaskedDecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, device):
        super(MaskedDecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device)
        self.pos_ffn = PositionWiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class PositionalEncodingM(nn.Module):
    """Implement the fixed PE function."""

    def __init__(self, d_model, dropout, max_len=5000, device=None):
        super(PositionalEncodingM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        # Compute the positional encodings once in log space.
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x, mask):
        bs, sl = x.shape[0], x.shape[1]
        pos = torch.from_numpy(np.arange(sl)).repeat(bs).view(bs, sl).to(self.device)
        x = x + self.pos_emb(pos)
        # mask shape: [n, len, 1] [n, len, d_model]
        return self.dropout(x * torch.unsqueeze(mask, 2))


class MaskedDecoder(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff, d_k,
                 d_v, n_heads, n_layers, pad_index, device):
        super(MaskedDecoder, self).__init__()
        self.pad_index = pad_index
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(
            d_model=d_model,
            dropout=0)

        self.layers = []
        for _ in range(n_layers):
            decoder_layer = MaskedDecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(
                dec_inputs=dec_outputs,
                dec_self_attn_mask=dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        return dec_outputs, dec_self_attns


class BertModel(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff,
                 d_k, d_v, n_heads, n_layers, pad_index,
                 device):
        super(BertModel, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(
            d_model=d_model,
            dropout=0)
        self.seg_embed = nn.Embedding(2, d_model)

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

        self.pad_index = pad_index

        self.fc = nn.Linear(d_model, d_model)
        self.active1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)

        self.linear = nn.Linear(d_model, d_model)
        self.active2 = GELU()
        self.norm = nn.LayerNorm(d_model)

        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.weight = self.tok_embed.weight
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.tok_embed(input_ids) + self.seg_embed(segment_ids)
        output = self.pos_embed(output)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.pad_index)

        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        h_pooled = self.active1(self.fc(output[:, 0]))
        logits_clsf = self.classifier(h_pooled)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.active2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, logits_clsf, output


class GPTModel(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff,
                 d_k, d_v, n_heads, n_layers, pad_index,
                 device):
        super(GPTModel, self).__init__()
        self.decoder = MaskedDecoder(
            vocab_size=vocab_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=pad_index,
            device=device)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, dec_inputs):
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits, dec_self_attns


class Classifier(nn.Module):

    def __init__(self, vocab_size, d_model, d_ff,
                 d_k, d_v, n_heads, n_layers,
                 pad_index, device, num_classes):
        super(Classifier, self).__init__()
        self.encoder = Encoder(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff,
                               d_k=d_k, d_v=d_v, n_heads=n_heads, n_layers=n_layers,
                               pad_index=pad_index, device=device)
        self.projection = nn.Linear(d_model, num_classes)

    def forward(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        mean_enc_outputs = torch.mean(enc_outputs, dim=1)
        logits = self.projection(mean_enc_outputs)
        return logits, enc_self_attns


class Translation(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers, src_pad_index,
                 tgt_pad_index, device):
        super(Translation, self).__init__()
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device)
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=tgt_pad_index,
            device=device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs, decode_lengths):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns, decode_lengths


class DecoderM(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device, dropout):
        super(DecoderM, self).__init__()
        self.pad_index = pad_index
        self.device = device
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads,
                device=device,
                dropout=dropout
            )
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs, enc_inputs, enc_outputs,
                dec_self_attn_mask, dec_enc_attn_mask, pad_m=None):
        dec_outputs = dec_inputs
        # dec_outputs = self.tgt_emb(dec_inputs)
        # dec_outputs = self.pos_emb(dec_outputs)
        #
        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        # dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        # dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.pad_index)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs,
                dec_self_attn_mask=dec_self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            # TODO add pad_mask.
            dec_outputs = dec_outputs * torch.unsqueeze(pad_m, 2)

            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_enc_attns = torch.stack(dec_enc_attns)

        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        dec_enc_attns = dec_enc_attns.permute([1, 0, 2, 3, 4])

        return dec_outputs, dec_self_attns, dec_enc_attns


def pad_mask_test():
    input_seq = [
        [1, 0, 0, 0, 0, 0],
        [1, 3, 0, 0, 0, 0],
        [1, 3, 4, 0, 0, 0],
        [1, 3, 4, 1, 0, 0]]
    src_emb = nn.Embedding(5, 5)
    input_seq = torch.as_tensor(input_seq, dtype=torch.long).to(torch.device('cpu'))
    # input_seq = src_emb(input_seq)
    pad_mask = get_attn_pad_mask(input_seq, input_seq, 0)
    future_mask = get_attn_subsequent_mask(input_seq)
    print("original input sequence: ==========", input_seq.size())
    print(input_seq)
    print("pad mask: ==========", pad_mask.size())
    print(pad_mask)
    print("future mask: ==========", future_mask.size())
    print(future_mask)


class EncoderM(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads, n_layers, pad_index, device, dropout):
        super(EncoderM, self).__init__()
        self.device = device
        self.pad_index = pad_index

        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device, dropout=dropout)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, enc_self_attn_mask, pad_mask=None):
        enc_outputs = x
        enc_self_attns = []
        for layer in self.layers:
            # print(enc_outputs[0, :])
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            # print(enc_outputs[0, :])
            enc_outputs = enc_outputs * torch.unsqueeze(pad_mask, 2)
            enc_self_attns.append(enc_self_attn)

        enc_self_attns = torch.stack(enc_self_attns)
        enc_self_attns = enc_self_attns.permute([1, 0, 2, 3, 4])
        return enc_outputs, enc_self_attns


class MaskedDecoderM(nn.Module):

    def __init__(self, d_model, d_ff, d_k,
                 d_v, n_heads, n_layers, pad_index, device):
        super(MaskedDecoderM, self).__init__()
        self.pad_index = pad_index
        self.layers = []

        for _ in range(n_layers):
            decoder_layer = MaskedDecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs = dec_inputs
        # dec_outputs = self.tgt_emb(dec_inputs)
        # dec_outputs = self.pos_emb(dec_outputs)

        # dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.pad_index)
        # dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(
                dec_inputs=dec_outputs,
                dec_self_attn_mask=dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        dec_self_attns = torch.stack(dec_self_attns)
        dec_self_attns = dec_self_attns.permute([1, 0, 2, 3, 4])
        return dec_outputs, dec_self_attns


class seq_recom(nn.Module):

    def __init__(self):
        super(seq_recom, self).__init__()


if __name__ == '__main__':
    pass
    # enc_input = [
    #     [1, 3, 4, 1, 2, 3],
    #     [1, 3, 4, 1, 2, 3],
    #     [1, 3, 4, 1, 2, 3],
    #     [1, 3, 4, 1, 2, 3]]
    # dec_input = [
    #     [1, 0, 0, 0, 0, 0],
    #     [1, 3, 0, 0, 0, 0],
    #     [1, 3, 4, 0, 0, 0],
    #     [1, 3, 4, 1, 0, 0]]
    # pad_mask_test()
    # enc_input = torch.as_tensor(enc_input, dtype=torch.long).to(torch.device('cpu'))
    # dec_input = torch.as_tensor(dec_input, dtype=torch.long).to(torch.device('cpu'))
    # model = Translation(
    #     src_vocab_size=5, tgt_vocab_size=5, d_model=128,
    #     d_ff=256, d_k=64, d_v=64, n_heads=8, n_layers=4, src_pad_index=0,
    #     tgt_pad_index=0, device=torch.device('cpu'))
    # logits, _, _, _, _ = model(enc_input, dec_input, 6)
    # print(logits)
    # enc_input = [
    #     [1, 3, 4, 1, 2, 3, 5],
    #     [1, 3, 4, 1, 2, 3, 5],
    #     [1, 3, 4, 1, 2, 3, 5],
    #     [1, 3, 4, 1, 2, 3, 5]]
    # dec_input = [
    #     [3, 4, 1, 2, 3],
    #     [3, 4, 1, 2, 3],
    #     [3, 4, 1, 2, 3],
    #     [3, 4, 1, 2, 3]]
    # # pad_mask_test()
    # enc_input = torch.as_tensor(enc_input, dtype=torch.long).to(torch.device('cpu'))
    # dec_input = torch.as_tensor(dec_input, dtype=torch.long).to(torch.device('cpu'))
    # model = Auto4rec(
    #     vocab_size=6, d_model=64, pad_index=0,
    #     d_ff=256, d_k=64, d_v=64, n_heads=4, n_layers=2, device=torch.device('cpu'))
    # # vocab_size, d_model, pad_index, d_ff, d_k, d_v, n_heads, n_layers, tgt_pad_index, device
    # enc_embedding, _, _, _, _ = model(enc_input, dec_input, 6)
    # print(enc_embedding)
