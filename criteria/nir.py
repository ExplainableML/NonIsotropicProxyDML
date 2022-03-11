import contextlib
import copy
import os

import clip
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import batchminer
"""================================================================================================="""
ALLOWED_MINING_OPS = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM = True


class Criterion(torch.nn.Module):
    def __init__(self, opt, **kwargs):
        """
        Args:
            opt: argparse.Namespace with all training parameters.
        """
        super(Criterion, self).__init__()
        self.pars = opt
        self.n_classes = opt.n_classes
        self.name = 'NIR'

        ####
        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

        ####
        self.nf_pos_alpha = opt.loss_nir_pos_alpha
        self.nf_neg_alpha = opt.loss_nir_neg_alpha
        self.nf_margin = opt.loss_nir_margin
        self.nf_delta = opt.loss_nir_delta
        self.nf_log_match = opt.loss_nir_logmatch

        ####
        self.optim_dict_list = []
        self.T = torch.Tensor([opt.language_temp])
        self.T = self.T.to(opt.device)

        self.proxies = None
        self.num_proxies = opt.n_classes
        self.embed_dim = opt.embed_dim

        self.proxies = torch.randn(self.num_proxies, self.embed_dim) / 8
        self.proxies = torch.nn.Parameter(self.proxies)

        self.cond_mode = opt.loss_nir_nf_cond_mode

        self.normflow = NormFlow(input_dim=opt.embed_dim,
                                 cond_mode=opt.loss_nir_nf_cond_mode,
                                 clamp=opt.loss_nir_nf_clamp_alpha,
                                 num_cblocks=opt.loss_nir_nf_cblocks,
                                 fc_depth=opt.loss_nir_nf_fc_depth,
                                 fc_width=opt.loss_nir_nf_fc_width,
                                 fc_dropout=opt.loss_nir_nf_fc_dropout,
                                 fc_init=opt.loss_nir_nf_fc_init,
                                 is_conditional=True)

        self.normflow_cond_prep = NormflowCondPrep(
            opt.embed_dim,
            self.normflow.num_cblocks,
            mode=self.normflow.cond_mode)

        self.pair_perc = self.pars.loss_nir_pair_perc
        self.noise = self.pars.loss_nir_noise
        self.w_align = self.pars.loss_nir_w_align

        self.iter_count = 0

        self.optim_dict_list = [{
            'params': self.proxies,
            'lr': opt.lr * opt.loss_nir_proxy_lrmulti
        }, {
            'params': self.normflow.parameters(),
            'lr': opt.lr * opt.loss_nir_lrmulti
        }, {
            'params': self.normflow_cond_prep.parameters(),
            'lr': opt.lr * opt.loss_nir_lrmulti
        }]

    def forward(self, batch, labels, avg_batch_features, **kwargs):
        """
        Args:
            batch: torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a
                    class [0,...,C-1], shape: (BS x 1)
        """
        self.iter_count += 1
        bdiff_labels = (labels.T != labels.view(-1, 1)).to(batch.device).T
        bsame_labels = (labels.T == labels.view(-1, 1)).to(batch.device).T

        # Base Proxy Objective for global proxy alignment.
        proxy_align_loss = self.panc(batch, self.proxies, labels)

        proxies = torch.nn.functional.normalize(self.proxies, dim=-1)
        proxies = proxies[labels]
        mul = np.ones(len(proxies))

        if isinstance(mul, np.ndarray):
            mul = torch.from_numpy(mul).to(batch.device)

        cond = self.normflow_cond_prep(proxies)
        if self.noise:
            batch = batch + self.noise * torch.rand(*batch.shape).to(
                batch.device)
            batch = torch.nn.functional.normalize(batch)

        nf_latent, log_jac_det = self.normflow(batch, cond=cond)

        loss = mul * (0.5 * torch.sum(nf_latent**2, dim=(1, )) - log_jac_det)

        loss = torch.mean(torch.exp(loss)) + self.w_align * proxy_align_loss

        return loss

    def pnca(self, batch, proxies, labels, dim=1):
        return self.panc(batch, proxies, labels, dim)

    def panc(self, batch, proxies, labels, dim=0):
        proxies = torch.nn.functional.normalize(proxies, dim=-1)
        batch = torch.nn.functional.normalize(batch, dim=-1)

        labels = labels.unsqueeze(1)
        u_labels, freq = labels.view(-1), None
        same_labels = (labels.T == u_labels.view(-1, 1)).to(batch.device).T
        diff_labels = (torch.arange(len(proxies)).unsqueeze(1) != labels.T).to(
            torch.float).to(batch.device).T

        w_pos_sims = -self.pars.loss_oproxy_pos_alpha * (
            batch.mm(proxies[u_labels].T) - self.pars.loss_oproxy_pos_delta)
        w_neg_sims = self.pars.loss_oproxy_neg_alpha * (
            batch.mm(proxies.T) - self.pars.loss_oproxy_neg_delta)

        pos_s = self.masked_logsumexp(w_pos_sims,
                                      mask=same_labels.type(torch.bool),
                                      dim=dim)
        neg_s = self.masked_logsumexp(w_neg_sims,
                                      mask=diff_labels.type(torch.bool),
                                      dim=dim)
        return pos_s.mean() + neg_s.mean()

    @staticmethod
    def masked_logsumexp(sims, dim=0, mask=None):
        # Adapted from https://github.com/KevinMusgrave/pytorch-metric-learning/\
        # blob/master/src/pytorch_metric_learning/utils/loss_and_miner_utils.py.
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape)
        dims[dim] = 1
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(
                ~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims


import FrEIA.framework as Ff
import FrEIA.modules as Fm


class NormFlow(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 cond_mode='dense',
                 num_cblocks=5,
                 fc_depth=8,
                 fc_width=128,
                 clamp=2.,
                 fc_dropout=0.,
                 fc_activation=torch.nn.ReLU,
                 is_conditional=True,
                 fc_init='none'):
        super(NormFlow, self).__init__()
        self.name = 'NormFlow'

        self.cond_mode = cond_mode
        self.num_cblocks = num_cblocks
        self.fc_depth = fc_depth
        self.fc_width = fc_width
        self.clamp = clamp
        self.fc_dropout = fc_dropout
        self.fc_activation = fc_activation
        self.is_conditional = is_conditional
        self.fc_init = fc_init

        self.normflow = self.init_normflow(input_dim)

        num_params = sum(p.numel() for p in self.normflow.parameters()
                         if p.requires_grad)
        print('-- Initialized NormFlow with {} params --'.format(num_params))

    def subnet_fc(self,
                  dims_in,
                  dims_out,
                  fc_depth=3,
                  fc_width=512,
                  dropout=0.,
                  activation=torch.nn.ReLU):
        layers = [torch.nn.Linear(dims_in, fc_width)]
        for _ in range(fc_depth - 2):
            layers.append(activation())
            lin_layer = torch.nn.Linear(fc_width, fc_width)
            layers.append(lin_layer)
            if self.fc_init == 'xavier':
                _ = torch.nn.init.xavier_uniform_(lin_layer.weight)
                _ = torch.nn.init.zeros_(lin_layer.bias)
        layers.append(activation())
        lin_layer = torch.nn.Linear(fc_width, dims_out)
        if self.fc_init == 'xavier':
            _ = torch.nn.init.xavier_uniform_(lin_layer.weight)
            _ = torch.nn.init.zeros_(lin_layer.bias)
        layers.append(lin_layer)
        return torch.nn.Sequential(*layers)

    def init_normflow(self, input_dim):
        cond_list = [None for _ in range(self.num_cblocks)]
        if self.is_conditional:
            if self.cond_mode == 'start':
                cond_list[0] = Ff.ConditionNode(input_dim, name='condition')
            if self.cond_mode == 'mid':
                cond_list[len(cond_list) // 2] = Ff.ConditionNode(
                    input_dim, name='condition')
            if self.cond_mode == 'end':
                cond_list[-1] = Ff.ConditionNode(input_dim, name='condition')
            if 'dense' in self.cond_mode:
                cond_list = [
                    Ff.ConditionNode(input_dim, name=F'condition_{k}')
                    for k in range(self.num_cblocks)
                ]

        node_list = []
        node_list.append(Ff.InputNode(input_dim, name='Input'))

        for k in range(self.num_cblocks):
            node_list.append(
                Ff.Node(node_list[-1],
                        Fm.PermuteRandom, {'seed': k},
                        name=F'permute_{k}'))
            node_list.append(
                Ff.Node(node_list[-1],
                        Fm.GLOWCouplingBlock, {
                            'clamp':
                            self.clamp,
                            'subnet_constructor':
                            lambda a, b: self.subnet_fc(
                                a, b, self.fc_depth, self.fc_width, self.
                                fc_dropout, self.fc_activation),
                        },
                        name=F'fc_{k}',
                        conditions=cond_list[k]))
        node_list.append(Ff.OutputNode(node_list[-1], name='output'))
        node_list += [x for x in cond_list if x is not None]
        return Ff.GraphINN(node_list)

    def forward(self, input, cond=None, rev=False, jac=True):
        return self.normflow(input, c=cond, rev=rev, jac=jac)


class NormflowCondPrep(torch.nn.Module):
    def __init__(self, embed_dim, num_blocks=5, mode='dense_embed'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.mode = mode

        if mode == 'dense_embed':
            self.embedder = torch.nn.Sequential(
                torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(),
                torch.nn.Linear(embed_dim, embed_dim))
        elif mode == 'dense_embed_multi' or mode == 'dense_embed_multi_stacked':
            self.embedder = torch.nn.ModuleList([
                torch.nn.Sequential(torch.nn.Linear(embed_dim, embed_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(embed_dim, embed_dim))
                for _ in range(self.num_blocks)
            ])

    def forward(self, x):
        if self.mode == 'dense_embed':
            x = self.embedder(x)
            return [x for _ in range(self.num_blocks)]
        elif self.mode == 'dense':
            out = []
            for _ in range(self.num_blocks):
                out.append(x)
            return out
        elif self.mode == 'dense_embed_multi':
            return [sub_embedder(x) for sub_embedder in self.embedder]
        elif self.mode == 'dense_embed_multi_stacked':
            out = []
            for sub_embedder in self.embedder:
                x = sub_embedder(x)
                out.append(x)
            return out
        else:
            return [x]
