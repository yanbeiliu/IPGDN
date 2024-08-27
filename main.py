#!/usr/bin/env python3
#
# See sample_results.ipynb for some sample results.
#
import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

import misc

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
matplotlib.use('TkAgg')
from sklearn import metrics


class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class DataReader:
    def __init__(self, data_name, data_dir):
        # Reading the data...
        tmp = []
        prefix = os.path.join(data_dir, 'ind.%s.' % data_name)
        for suffix in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open(prefix + suffix, 'rb') as fin:
                tmp.append(pickle.load(fin, encoding='latin1'))
        x, y, tx, ty, allx, ally, graph = tmp
        with open(prefix + 'test.index') as fin:
            tst_idx = [int(i) for i in fin.read().split()]
        assert np.sum(x != allx[:x.shape[0], :]) == 0
        assert np.sum(y != ally[:y.shape[0], :]) == 0

        # Spliting the data...
        trn_idx = np.array(range(x.shape[0]), dtype=np.int64)
        val_idx = np.array(range(x.shape[0], allx.shape[0]), dtype=np.int64)
        tst_idx = np.array(tst_idx, dtype=np.int64)
        assert len(trn_idx) == x.shape[0]
        assert len(trn_idx) + len(val_idx) == allx.shape[0]
        assert len(tst_idx) > 0
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0

        # Building the graph...
        graph = nx.from_dict_of_lists(graph)
        assert min(graph.nodes()) == 0
        n = graph.number_of_nodes()
        assert max(graph.nodes()) + 1 == n
        n = max(n, np.max(tst_idx) + 1)
        for u in range(n):
            graph.add_node(u)
        assert graph.number_of_nodes() == n
        assert not graph.is_directed()

        # Building the feature matrix and the label matrix...
        d, c = x.shape[1], y.shape[1]
        feat_ridx, feat_cidx, feat_data = [], [], []
        allx_coo = allx.tocoo()
        for i, j, v in zip(allx_coo.row, allx_coo.col, allx_coo.data):
            feat_ridx.append(i)
            feat_cidx.append(j)
            feat_data.append(v)
        tx_coo = tx.tocoo()
        for i, j, v in zip(tx_coo.row, tx_coo.col, tx_coo.data):
            feat_ridx.append(tst_idx[i])
            feat_cidx.append(j)
            feat_data.append(v)
        if data_name.startswith('nell.0'):
            isolated = np.sort(np.setdiff1d(range(allx.shape[0], n), tst_idx))
            for i, r in enumerate(isolated):
                feat_ridx.append(r)
                feat_cidx.append(d + i)
                feat_data.append(1)
            d += len(isolated)
        feat = spsprs.csr_matrix((feat_data, (feat_ridx, feat_cidx)), (n, d))
        targ = np.zeros((n, c), dtype=np.int64)
        targ[trn_idx, :] = y
        targ[val_idx, :] = ally[val_idx, :]
        targ[tst_idx, :] = ty
        targ = dict((i, j) for i, j in zip(*np.where(targ)))
        targ = np.array([targ.get(i, -1) for i in range(n)], dtype=np.int64)
        print('#instance x #feature ~ #class = %d x %d ~ %d' % (n, d, c))

        # Storing the data...
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.graph, self.feat, self.targ = graph, feat, targ

    def get_split(self):
        # *val_idx* contains unlabeled samples for semi-supervised training.
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ


# noinspection PyUnresolvedReferences
def thsprs_from_spsprs(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))


# noinspection PyUnresolvedReferences
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias


# noinspection PyUnresolvedReferences
class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


# noinspection PyUnresolvedReferences
class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps):
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, delta_d = self.d, self.k, self.d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)
        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)


class CapsuleNet(nn.Module):  # CapsuleNet = DisenGCN
    def __init__(self, nfeat, nclass, hyperpm):
        super(CapsuleNet, self).__init__()
        ncaps, rep_dim = hyperpm.ncaps, hyperpm.nhidden * hyperpm.ncaps
        self.pca = SparseInputLinear(nfeat, rep_dim)
        conv_ls = []
        for i in range(hyperpm.nlayer):
            conv = RoutingLayer(rep_dim, ncaps)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.mlp = nn.Linear(rep_dim, nclass)
        self.dropout = hyperpm.dropout
        self.routit = hyperpm.routit

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb):
        nb = nb.view(-1)
        x = fn.relu(self.pca(x))       

        # new codding
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        x_f = self.mlp(x)
        return fn.log_softmax(x_f, dim=1), x


# noinspection PyUnresolvedReferences
def real2col(x):
    assert 0.0 <= x <= 1.0
    r, g, b, a = matplotlib.cm.gist_ncar(x)
    return '%d,%d,%d' % (r * 255, g * 255, b * 255)


def visualize_as_gdf(g, savfile, label, color, pos_gml=None):
    assert isinstance(g, nx.Graph)
    n = g.number_of_nodes()
    if not savfile.endswith('.gdf'):
        savfile += '.gdf'
    assert len(label) == n
    color = np.asarray(color, dtype=np.float32).copy()
    color = (color - color.min()) / (color.max() - color.min() + 1e-6)
    assert color.shape == (n,)
    if isinstance(pos_gml, str) and os.path.isfile(pos_gml):
        layout_g = nx.read_gml(pos_gml)
        layout_g = dict(layout_g.nodes)
        pos = np.zeros((n, 2), dtype=np.float64)
        for t in range(n):
            pos[t] = (layout_g[str(t)]['graphics']['x'],
                      layout_g[str(t)]['graphics']['y'])
        scale = 1
    else:
        pos = nx.random_layout(g)
        scale = 1000
    with open(savfile, 'w') as fout:
        fout.write('nodedef>name VARCHAR,label VARCHAR,'
                   'x DOUBLE,y DOUBLE,color VARCHAR\n')
        for t in range(n):
            fout.write("%d,%s,%f,%f,'%s'\n" %
                       (t, label[t], pos[t][0] * scale, pos[t][1] * scale,
                        real2col(color[t])))
        fout.write('edgedef>node1 VARCHAR,node2 VARCHAR\n')
        for (u, v) in g.edges():
            fout.write('%d,%d\n' % (u, v))


class EvalHelper:
    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, hyperpm):
        use_cuda = torch.cuda.is_available() and not hyperpm.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        graph, feat, targ = dataset.get_graph_feat_targ()
        targ = torch.from_numpy(targ).to(dev)
        feat = thsprs_from_spsprs(feat).to(dev)
        trn_idx, val_idx, tst_idx = dataset.get_split()
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        nfeat, nclass = feat.size(1), int(targ.max() + 1)
        model = CapsuleNet(nfeat, nclass, hyperpm).to(dev)
        optmz = optim.Adam(model.parameters(),
                           lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.graph, self.feat, self.targ = graph, feat, targ
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz = model, optmz
        self.neib_sampler = NeibSampler(graph, hyperpm.nbsz).to(dev)
        self.ncaps = hyperpm.ncaps  # new codding
        self.nhidden = hyperpm.nhidden
        self.lamda = hyperpm.lamda

    def run_epoch(self, end='\n'):
        self.model.train()
        self.optmz.zero_grad()        

        # loss function by new codding
        prob, zdata = self.model(self.feat, self.neib_sampler.sample())

        loss_c = fn.nll_loss(prob[self.trn_idx], self.targ[self.trn_idx])       
        loss_dep = loss_dependence(zdata[self.trn_idx], self.ncaps, self.nhidden)
        loss = loss_c + self.lamda*loss_dep                                              # new codding

        loss.backward()
        self.optmz.step()
        print('trn-loss: %.4f' % loss.item(), end=end)
        return loss.item()

    def print_trn_acc(self):
        print('trn-', end='')
        trn_acc = self._print_acc(self.trn_idx, end=' val-')
        val_acc = self._print_acc(self.val_idx)
        return trn_acc, val_acc

    def print_tst_acc(self):
        print('tst-', end='')
        tst_acc = self._print_acc(self.tst_idx)
        return tst_acc

    def _print_acc(self, eval_idx, end='\n'):
        self.model.eval()      
        # new codding
        prob, zdata = self.model(self.feat, self.neib_sampler.nb_all)
        prob = prob[eval_idx]

        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        # # ACC
        
        # F1 score
        y_true = targ.cpu().detach().numpy()
        y_pre = pred.cpu().detach().numpy()
        f1f1 = metrics.f1_score(y_true, y_pre, average='weighted')
        acc = f1f1

        print('acc: %.4f' % acc, end=end)
        return acc

    # compute mask zdata clustering   by new codding
    def measure_clustering(self):
        self.model.eval()
        prob, zdata = self.model(self.feat, self.neib_sampler.nb_all)
        zdata = zdata[self.tst_idx].cpu().detach().numpy()
        targ = self.targ[self.tst_idx].cpu().numpy()
        nclass = targ.max() + 1
        
        return misc.evaluateKMeans(zdata, targ, nclass)


# noinspection PyUnresolvedReferences
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def loss_dependence(zdata_trn, ncaps, nhidden):
    loss_dep = torch.zeros(1).cuda()
    hH = (-1/nhidden)*torch.ones(nhidden, nhidden).cuda() + torch.eye(nhidden).cuda()
    kfactor = torch.zeros(ncaps, nhidden, nhidden).cuda()

    for mm in range(ncaps):
        data_temp = zdata_trn[:, mm * nhidden:(mm + 1) * nhidden]
        kfactor[mm, :, :] = torch.mm(data_temp.t(), data_temp)

    for mm in range(ncaps):
        for mn in range(mm + 1, ncaps):
            mat1 = torch.mm(hH, kfactor[mm, :, :])
            mat2 = torch.mm(hH, kfactor[mn, :, :])
            mat3 = torch.mm(mat1, mat2)
            teststat = torch.trace(mat3)

            loss_dep = loss_dep + teststat
    return loss_dep


# noinspection PyUnresolvedReferences
def train_and_eval(datadir, datname, hyperpm):
    set_rng_seed(23)
    agent = EvalHelper(DataReader(datname, datadir), hyperpm)
    tm = time.time()
    best_val_acc, wait_cnt = 0.0, 0
    # loss line
    loss_train_my = torch.zeros(hyperpm.nepoch)

    model_sav = tempfile.TemporaryFile()
    neib_sav = torch.zeros_like(agent.neib_sampler.nb_all, device='cpu')
    for t in range(hyperpm.nepoch):
        print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
        # agent.run_epoch(end=' ')
        loss_train_my[t] = agent.run_epoch(end=' ')
        _, cur_val_acc = agent.print_trn_acc()
        if cur_val_acc > best_val_acc:
            wait_cnt = 0
            best_val_acc = cur_val_acc
            model_sav.close()
            model_sav = tempfile.TemporaryFile()
            torch.save(agent.model.state_dict(), model_sav)
            neib_sav.copy_(agent.neib_sampler.nb_all)
        else:
            wait_cnt += 1
            if wait_cnt > hyperpm.early:
                break
    print("time: %.4f sec." % (time.time() - tm))
    model_sav.seek(0)
    agent.model.load_state_dict(torch.load(model_sav))
    agent.neib_sampler.nb_all.copy_(neib_sav)    

    return best_val_acc, agent.print_tst_acc()


def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='Cora')   # Cora, Citeseer, Pubmed
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=200,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=8,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=0.17,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.00036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.32,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=5,
                        help='Number of conv layers.')
    parser.add_argument('--ncaps', type=int, default=5,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=18,
                        help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=7,
                        help='Number of iterations when routing.')
    parser.add_argument('--nbsz', type=int, default=30,
                        help='Size of the sampled neighborhood.')
    parser.add_argument('--lamda', type=float, default=8e-6,
                        help='weight of dependence.')
    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with RedirectStdStreams(stdout=sys.stderr):
        val_acc, tst_acc = train_and_eval(args.datadir, args.datname, args)
        print('val=%.2f%% tst=%.2f%%' % (val_acc * 100, tst_acc * 100))
    return val_acc, tst_acc


if __name__ == '__main__':
    print('(%.4f, %.4f)' % main())
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
