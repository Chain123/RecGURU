# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, loss_type='TOP1'):
        """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'CrossEntropy':
            self._loss_fn = SampledCrossEntropyLoss()
        elif loss_type == 'TOP1':
            self._loss_fn = TOP1Loss()
        elif loss_type == 'BPR':
            self._loss_fn = BPRLoss()
        elif loss_type == 'TOP1-max':
            self._loss_fn = TOP1_max()
        elif loss_type == 'BPR-max':
            self._loss_fn = BPR_max()
        else:
            raise NotImplementedError

    def forward(self, logit):
        return self._loss_fn(logit)


class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """

    def __init__(self, reduction="none"):
        super(SampledCrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logit, label, num_class, mask=None):
        """
        :param mask:
        :param num_class:
        :param label:
        :param logit: all_logits, (positive and negative.)
        :return: cross-entropy loss
        """
        loss = self.xe_loss(logit.view(-1, num_class), label)
        if mask is not None:
            loss_m = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
        else:
            loss_m = torch.mean(loss)
        return loss_m


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, p_logit, n_logit, mask=None):
        """
        :param mask:
        :param p_logit:  logit for positive sample: [b, len, 1]
        :param n_logit:  logits for negative sample [b, len, neg]
        :return:         BPR loss
        """
        # differences between the item scores
        p_logit = p_logit.view(-1)
        n_logit = torch.mean(n_logit, 2).view(-1)
        loss = -(p_logit - n_logit).sigmoid().log()
        # final loss
        if mask is not None:
            loss_m = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
        else:
            loss_m = torch.mean(loss)
        return loss_m


class BPRLoss_sas(nn.Module):
    def __init__(self):
        super(BPRLoss_sas, self).__init__()

    def forward(self, p_logit, n_logit, mask=None):
        """
        :param mask:
        :param p_logit:  logit for positive sample: [b, len, 1]
        :param n_logit:  logits for negative sample [b, len, neg]
        :return:         BPR loss
        """
        # differences between the item scores
        p_logit = p_logit.view(-1)
        n_logit = torch.mean(n_logit, 2).view(-1)
        # loss = -(p_logit - n_logit).sigmoid().log()
        loss = - ((p_logit.sigmoid() + 1e-24).log() + (1 - n_logit.sigmoid() + 1e-24).log())
        # final loss
        if mask is not None:
            loss_m = torch.sum(torch.mul(loss, mask)) / torch.sum(mask)
        else:
            loss_m = torch.mean(loss)
        return loss_m


class BPR_max(nn.Module):
    def __init__(self):
        super(BPR_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = logit.diag().view(-1, 1).expand_as(logit) - logit
        loss = -torch.log(torch.mean(logit_softmax * torch.sigmoid(diff)))
        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, logit):
        """
        Args:
            logit (BxB): Variable that stores the logits for the items in the mini-batch
                         The first dimension corresponds to the batches, and the second
                         dimension corresponds to sampled number of items to evaluate
        """
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.sigmoid(diff).mean() + torch.sigmoid(logit ** 2).mean()
        return loss


class TOP1_max(nn.Module):
    def __init__(self):
        super(TOP1_max, self).__init__()

    def forward(self, logit):
        logit_softmax = F.softmax(logit, dim=1)
        diff = -(logit.diag().view(-1, 1).expand_as(logit) - logit)
        loss = torch.mean(logit_softmax * (torch.sigmoid(diff) + torch.sigmoid(logit ** 2)))
        return loss
