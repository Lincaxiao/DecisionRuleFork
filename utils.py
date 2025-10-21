import torch
import torch.nn as nn
import numpy as np
import models
from problem import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import json
import math
import random
import time
import csv
import os


def load_problem(args):
    if args.problem == "primal_lp":
        c = torch.load(f'./data/{args.dataset}/new_feasibility/c_backbone.pt').to(args.device)
        nonnegative_mask = torch.load(f'./data/{args.dataset}/new_feasibility/nonnegative_mask.pt')
        if args.algo == 'DC3LHS':
            b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt').to(args.device)
            A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt').to(args.device)
            DC3LHS = True
            A_eq = A_backbone[:args.eq_constr_num, :(args.var_num - args.x_nneg_num)]
            b_eq = b_backbone[:args.eq_constr_num]
            h_ineq = b_backbone[args.eq_constr_num:]
        else:
            DC3LHS = False
            A_eq = None
            b_eq = None
            h_ineq = None
        problem = PrimalLP(c=c, nonnegative_mask=nonnegative_mask, DC3LHS=DC3LHS,
                           A_eq=A_eq, b_eq=b_eq, h_ineq=h_ineq)
    else:
        raise ValueError('Invalid problem')
    return problem


def load_model(args):

    act_cls_mapping = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU,
                       'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softmax': nn.Softmax}

    mlp = models.MLP(features_dim=args.feature_dim,
                     hidden_dims=args.hidden_dims,
                     out_dim=args.out_dim,
                     act_cls=act_cls_mapping[args.act_cls],
                     batch_norm=args.batch_norm)

    return mlp.to(args.device)


def load_algo(args):
    nonnegative_mask = torch.load(f'./data/{args.dataset}/new_feasibility/nonnegative_mask.pt').to(args.device)

    b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt').to(args.device)
    A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt').to(args.device)

    if args.algo == 'LDRPM':
        ldr_weight = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_weight.pt').to(args.device)
        ldr_bias = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_bias.pt').to(args.device)
        eq_weight, eq_bias_transform = compute_eq_projector(A_backbone)
        algo = models.LDRPM(nonnegative_mask=nonnegative_mask,
                            eq_weight=eq_weight, eq_bias_transform=eq_bias_transform,
                            ldr_weight=ldr_weight, ldr_bias=ldr_bias, ldr_temp=args.ldr_temp)

    elif args.algo == 'POCS':
        eq_weight, eq_bias_transform = compute_eq_projector(A_backbone)
        algo = models.POCS(nonnegative_mask=nonnegative_mask.bool(),
                           eq_weight=eq_weight, eq_bias_transform=eq_bias_transform)
    elif args.algo == 'POCSLHS':
        eq_weight, eq_bias_transform = compute_eq_projector(A_backbone)
        algo = models.POCS(nonnegative_mask=nonnegative_mask.bool(),
                           eq_weight=eq_weight, eq_bias_transform=eq_bias_transform)
        algo.name = 'POCSLHS'

    elif args.algo == 'DC3':
        algo = models.DC3(A=A_backbone, nonnegative_mask=nonnegative_mask,
                          lr=args.dc3_lr, momentum=args.dc3_momentum,
                          changing_feature=args.changing_feature)
    elif args.algo == 'DC3LHS':
        Aeq_backbone = A_backbone[:args.eq_constr_num, :(args.var_num - args.x_nneg_num)]
        beq_backbone = b_backbone[:args.eq_constr_num]
        hineq_backbone = b_backbone[args.eq_constr_num:]
        algo = models.DC3LHS(A=Aeq_backbone, nonnegative_mask=nonnegative_mask, h=hineq_backbone, b=beq_backbone,
                             lr=args.dc3_lr, momentum=args.dc3_momentum)

    elif args.algo == 'LDRPMLHS':
        ldr_weight = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_weight.pt').to(args.device)
        ldr_bias = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_bias.pt').to(args.device)
        S_scale = torch.load(f'./data/{args.dataset}/new_feasibility/S_scale.pt').view(args.feature_num + 1, -1,
                                                                                       args.feature_num + 1).permute(1,
                                                                                                                     0,
                                                                                                                     2).to(args.device)
        # A_backbone -> eq backbone
        Aeq_backbone = A_backbone[:args.eq_constr_num, :(args.var_num - args.x_nneg_num)]
        beq_backbone = b_backbone[:args.eq_constr_num]
        hineq_backbone = b_backbone[args.eq_constr_num:]
        eq_weight, eq_bias_transform = compute_eq_projector(Aeq_backbone)
        eq_bias = eq_bias_transform @ beq_backbone

        algo = models.LDRPMLHS(h=hineq_backbone,
                               eq_weight=eq_weight, eq_bias=eq_bias, ldr_weight=ldr_weight, ldr_bias=ldr_bias,
                               S_scale=S_scale, nonnegative_mask=nonnegative_mask)

    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    algo = algo.to(args.device)
    print(f"Loaded algorithm: {args.algo}")
    return models.FeasibilityNet(algo=algo,
                                 eq_tol=args.eq_tol, ineq_tol=args.ineq_tol, max_iters=args.max_iters,
                                 changing_feature=args.changing_feature).to(args.device)


def compute_eq_projector(A):
    A = A.to_sparse() if not A.is_sparse else A
    with torch.no_grad():
        PD = torch.sparse.mm(A, A.t())
        chunk = torch.sparse.mm(A.t(), torch.inverse(PD.to_dense()))
        eq_weight = torch.eye(A.shape[-1]).to(A.device) - torch.sparse.mm(chunk, A)
        eq_bias_transform = chunk
    return eq_weight, eq_bias_transform


def load_instances(args, b_scale, A_scale, b_backbone, A_backbone, train_val_test):
    features = torch.load(f'./data/{args.dataset}/new_feasibility/{train_val_test}/features.pt')
    features = torch.cat([torch.ones((len(features), 1)), features], dim=1)  # add bias term
    targets = torch.load(f'./data/{args.dataset}/new_feasibility/{train_val_test}/targets.pt')

    print(f'Loaded {train_val_test} data: {len(features)} instances')

    dataset = []
    for i in range(len(features)):
        feature = features[i]
        target = targets[i]
        if args.changing_feature == 'b':
            b = b_scale @ feature
            A = A_backbone.clone().detach()
        elif args.changing_feature == 'A':
            b = b_backbone
            A_scale_dense = A_scale.view(args.constr_num, -1, args.var_num)
            A = torch.einsum('d,dmn->mn', feature, A_scale_dense)
            G = A[args.eq_constr_num:, :(args.var_num - args.x_nneg_num)]
            A_eq = A[:args.eq_constr_num, :(args.var_num - args.x_nneg_num)]
        else:
            raise ValueError('Invalid changing_feature')

        A_sparse = A.to_sparse() if not A.is_sparse else A
        A_indices = A_sparse.indices()
        A_values = A_sparse.values()

        if args.changing_feature == 'b':
            dataset.append(BasicData(feature=feature[1:], target=target,
                                     b=b, A_indices=A_indices, A_values=A_values,
                                     constr_num=args.constr_num, var_num=args.var_num))
        elif args.changing_feature == 'A':
            dataset.append(BasicData(feature=feature[1:], target=target,
                                     b=b, A_indices=A_indices, A_values=A_values,
                                     constr_num=args.constr_num, var_num=args.var_num,
                                     G=G, A_eq=A_eq,
                                     ineq_constr_num=args.ineq_constr_num))
        else:
            raise ValueError('Invalid changing_feature')

    if train_val_test == 'train':
        bsz = args.batch_size
        shuffle = True
    elif train_val_test == 'val':
        bsz = args.batch_size
        shuffle = False
    elif train_val_test == 'test':
        bsz = 1
        shuffle = False
    else:
        raise ValueError('Invalid train_val_test')
    return DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=0, pin_memory=False, persistent_workers=False)


def load_data(args):
    b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt')
    A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt')

    if args.changing_feature == 'b':
        b_scale = torch.load(f'./data/{args.dataset}/new_feasibility/b_scale.pt')
        A_scale = None
        b_backbone = None
        A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt')
    elif args.changing_feature == 'A':
        b_scale = None
        A_scale = torch.load(f'./data/{args.dataset}/new_feasibility/A_scale.pt')
        b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt')
        A_backbone = None
    else:
        raise ValueError('Invalid changing_feature')

    if args.job in ['training']:
        if args.data_generator:
            train = DataLoader(
                InstanceDataset(args, b_scale, A_scale, b_backbone, A_backbone),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                exclude_keys=['constr_num', 'var_num']
            )
        else:
            train = load_instances(args, b_scale, A_scale,
                                      b_backbone, A_backbone, 'train')
        val = load_instances(args, b_scale, A_scale,
                                      b_backbone, A_backbone, 'val')
        test = load_instances(args, b_scale, A_scale,
                                      b_backbone, A_backbone, 'test')
        data = {'train': train, 'val': val, 'test': test}
        return data


class BasicData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'A_indices':
            return torch.tensor([[self.constr_num], [self.var_num]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['A_indices']:
            return 1
        if key in ['A_values']:
            return 0
        if key in ['feature', 'b', 'G', 'A_eq']:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, args, b_scale, A_scale, b_backbone, A_backbone):
        self.args          = args
        self.b_scale       = b_scale
        self.A_scale       = A_scale
        self.b_backbone    = b_backbone
        self.A_backbone    = A_backbone
        self.N             = args.bsz_factor * args.batch_size   # total samples/epoch

        self.feature_lb = torch.ones(1 + args.feature_num)
        self.feature_ub = torch.cat([torch.ones(1), -torch.ones(args.feature_num)])

        self._seed = torch.seed()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        feature = torch.rand_like(self.feature_lb) * (self.feature_ub - self.feature_lb) + self.feature_lb

        if self.args.changing_feature == "b":
            b = self.b_scale @ feature
            A = self.A_backbone
        elif self.args.changing_feature == "A":
            b = self.b_backbone
            A_scale_dense = self.A_scale.view(self.args.constr_num, -1, self.args.var_num)
            A = torch.einsum('i,ijk->jk', feature, A_scale_dense)

            G = A[self.args.eq_constr_num:, :(self.args.var_num - self.args.x_nneg_num)]
            A_eq = A[:self.args.eq_constr_num, :(self.args.var_num - self.args.x_nneg_num)]
        else:
            raise ValueError('Invalid changing_feature')

        A = A.to_sparse() if not A.is_sparse else A

        if self.args.changing_feature == "b":
            return BasicData(
                feature=feature[1:],
                target=torch.zeros(1),
                b=b,
                A_indices=A.indices(),
                A_values=A.values(),
                constr_num=self.args.constr_num,
                var_num=self.args.var_num,
            )
        elif self.args.changing_feature == "A":
            return BasicData(
                feature=feature[1:],
                target=torch.zeros(1),
                b=b,
                A_indices=A.indices(),
                A_values=A.values(),
                constr_num=self.args.constr_num,
                var_num=self.args.var_num,
                G=G,
                A_eq=A_eq,
                ineq_constr_num=self.args.ineq_constr_num,
            )
        else:
            raise ValueError('Invalid changing_feature')

    def refresh(self):
        self._seed = torch.seed()


def get_optimizer(args, model):
    params = model.parameters()
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer')
    return optimizer


def get_ldr_result(args):
    problem = load_problem(args)
    data = load_data(args)

    nonnegative_mask = torch.load(f'./data/{args.dataset}/new_feasibility/nonnegative_mask.pt').to(args.device)
    b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt').to(args.device)
    A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt').to(args.device)

    if args.algo == 'LDRPM':
        ldr_weight = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_weight.pt').to(args.device).t().requires_grad_(False)
        ldr_bias = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_bias.pt').to(args.device).requires_grad_(False)
        eq_weight, eq_bias_transform = compute_eq_projector(A_backbone)

        b_scale = torch.load(f'./data/{args.dataset}/new_feasibility/b_scale.pt')
        A_scale = None
        b_backbone = None
        A_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/A_backbone.pt')

    elif args.algo == 'LDRPMLHS':
        ldr_weight = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_weight.pt').to(args.device).t().requires_grad_(False)
        ldr_bias = torch.load(f'./data/{args.dataset}/new_feasibility/ldr_bias.pt').to(args.device).requires_grad_(False)
        S_scale = torch.load(f'./data/{args.dataset}/new_feasibility/S_scale.pt').view(args.feature_num + 1, -1,
                                                                                       args.feature_num + 1).permute(1,
                                                                                                                     0,
                                                                                                                     2).to(
            args.device)
        # A_backbone -> eq backbone
        Aeq_backbone = A_backbone[:args.eq_constr_num, :(args.var_num - args.x_nneg_num)]
        beq_backbone = b_backbone[:args.eq_constr_num]
        hineq_backbone = b_backbone[args.eq_constr_num:]
        eq_weight, eq_bias_transform = compute_eq_projector(Aeq_backbone)
        eq_bias = eq_bias_transform @ beq_backbone

        b_scale = None
        A_scale = torch.load(f'./data/{args.dataset}/new_feasibility/A_scale.pt')
        b_backbone = torch.load(f'./data/{args.dataset}/new_feasibility/b_backbone.pt')
        A_backbone = None

    else:
        raise ValueError('Invalid algorithm')

    test = load_instances(args, b_scale, A_scale, b_backbone, A_backbone, 'test')

    test_stats = {"test_time": 0,
                  "test_gap": 0,
                  "test_gap_worst": 0,
                  "test_agg": 0,
                  'test_iters': 0,
                  'test_eq': 0,
                  'test_ineq': 0,
                  'test_eq_worst': 0,
                  'test_ineq_worst': 0}

    with torch.no_grad():
        for i, batch in enumerate(test):
            start_time = time.time()

            bsz = batch.feature.shape[0]
            A_sp = torch.sparse_coo_tensor(indices=batch.A_indices,
                                           values=batch.A_values,
                                           size=(bsz * args.constr_num, bsz * args.var_num)).to(args.device)
            batch = batch.to(args.device)

            G = batch.G if hasattr(batch, 'G') else None
            A_eq = batch.A_eq if hasattr(batch, 'A_eq') else None

            iters = 0
            eq_part = ldr_bias + batch.feature @ ldr_weight
            if args.algo == 'LDRPMLHS':
                eq_part = ldr_bias + batch.feature @ ldr_weight
                ones_features = torch.cat((torch.ones(batch.feature.shape[0], 1, device=args.device), batch.feature), dim=1)
                ineq_part = torch.einsum('bd,ndm->bnm', ones_features, S_scale)
                ineq_part = torch.einsum('bnm,bm->bn', ineq_part, ones_features)
                x_LDR = torch.cat((eq_part, ineq_part), dim=1)

            elif args.algo == 'LDRPM':
                x_LDR = eq_part

            iters += 1

            predicted_obj = problem.obj_fn(x=x_LDR)
            optimality_gap = problem.optimality_gap(predicted_obj, batch.target).mean()

            test_time = time.time() - start_time

            eq_residual = (A_sp @ x_LDR.flatten() - batch.b.flatten()).view(-1, batch.b.shape[-1])
            ineq_residual = torch.relu(-x_LDR[:, problem.nonnegative_mask])
            eq_violation = torch.norm(eq_residual, p=2, dim=-1)
            ineq_violation = torch.norm(ineq_residual, p=2, dim=-1)
            eq_epsilon = eq_violation / (1 + torch.norm(batch.b, p=2, dim=-1))  # scaled by the norm of b
            ineq_epsilon = ineq_violation  # no scaling because rhs is 0

            gap = float(optimality_gap.mean().detach().cpu())
            gap_worst = float(torch.norm(optimality_gap, float('inf')).detach().cpu())  # max of samples
            test_stats["test_time"] += test_time
            test_stats["test_gap"] += gap
            test_stats["test_gap_worst"] = max(test_stats["test_gap_worst"], gap_worst)
            test_stats["test_agg"] += 1
            test_stats["test_iters"] += iters
            test_stats["test_eq"] += float(eq_epsilon.mean().detach().cpu())
            test_stats["test_ineq"] += float(ineq_epsilon.mean().detach().cpu())
            test_stats["test_eq_worst"] = max(test_stats["test_eq_worst"],
                                              float(torch.norm(eq_epsilon, float('inf')).detach().cpu()))
            test_stats["test_ineq_worst"] = max(test_stats["test_ineq_worst"],
                                                  float(torch.norm(ineq_epsilon, float('inf')).detach().cpu()))

    scores = {'test_optimality_gap_mean': test_stats['test_gap'] / test_stats['test_agg'],
              'test_optimality_gap_worst': test_stats['test_gap_worst'],
              'test_eq_violation_mean': test_stats['test_eq'] / test_stats['test_agg'],
              'test_eq_violation_worst': test_stats['test_eq_worst'],
              'test_ineq_violation_mean': test_stats['test_ineq'] / test_stats['test_agg'],
              'test_ineq_violation_worst': test_stats['test_ineq_worst'],
              'test_iters_mean': int(test_stats['test_iters'] // test_stats['test_agg']),
              'test_time': test_stats['test_time'] / test_stats['test_agg'], }
    print(scores)
    args_dict = vars(args)
    args_scores_dict = args_dict | scores
    w = csv.writer(open('./data/results_summary/' + f"{args.dataset}_feasibility_ldr" + '.csv', 'w'))
    for key, val in args_scores_dict.items():
        w.writerow([key, val])



