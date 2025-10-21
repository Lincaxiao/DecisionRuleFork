from utils import *
import torch
import torch.nn.functional as F
import csv
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import psutil


def log_cpu_memory_usage(epoch, step=None):
    # Get the current process
    process = psutil.Process(os.getpid())
    # Get memory info
    memory_info = process.memory_info()
    if epoch % 1 == 0:
        # Log RSS (Resident Set Size) in GB
        print(f"[Epoch {epoch}{f', Step {step}' if step is not None else ''}] CPU Memory - RSS: {memory_info.rss / (1024 ** 3):.2f} GB")


def run_training(args, data, problem):
    model = load_model(args)
    feasibility_net = load_algo(args)
    if args.continue_training:
        print('loading pretrained model weights')
        model = load_weights(model, args)
    print(f'----- {args.model_id} in {args.dataset} dataset -----')
    print('#params:', sum(p.numel() for p in model.parameters()))
    optimizer = get_optimizer(args, model)
    print('loss_type: ', args.loss_type)

    start_time = time.time()
    ##############################################################################################################
    Learning(args, data, problem, model, feasibility_net, optimizer)
    ##############################################################################################################
    end_time = time.time()
    training_time = end_time - start_time
    print(f'----time required for {args.epochs} epochs training: {round(training_time)}s----')
    print(f'----time required for {args.epochs} epochs training: {round(training_time / 60)}min----')
    print(f'----time required for {args.epochs} epochs training: {round(training_time / 3600)}hr----')
    # check the model on the test set
    scores = evaluate_model(args, data, problem)


def train_model(optimizer, model, feasibility_net, args, data, problem, epoch_stats):
    for batch in (data['train'] if args.data_generator else data['train']):
        optimizer_step(model, feasibility_net, optimizer, batch, args, problem, epoch_stats)


def Learning(args, data, problem, model, feasibility_net, optimizer):
    best = float('inf')
    stats = {}
    for epoch in range(args.pretrain_epochs + args.epochs):
        args.epoch = epoch
        if args.data_generator and epoch % args.renew_freq == 0:
            # data['train'].reset_data()
            data['train'].dataset.refresh()

        if args.algo == 'LDRPM':
            if epoch <= args.pretrain_epochs:
                # pretrain the model
                args.loss_type = 'mse'
            else:
                args.loss_type = 'obj'

        epoch_stats = {}
        # train
        model.train()
        feasibility_net.train()
        train_model(optimizer, model, feasibility_net, args, data, problem, epoch_stats)
        if epoch % 1 == 0 and 'LDRPM' in args.algo:
            print(f'epoch {epoch},  alpha: {feasibility_net.algo.alpha.mean()}')
        curr_loss = epoch_stats['train_loss'] / epoch_stats['train_agg']
        # log_cpu_memory_usage(epoch, 'training')

        # validate
        model.eval()
        feasibility_net.eval()
        validate_model(model, feasibility_net, args, data, problem, epoch_stats)
        # log_cpu_memory_usage(epoch, 'validation')

        # model checkpoint
        curr_gap = abs(epoch_stats['val_gap'] / epoch_stats['val_agg'])
        if curr_gap < best:
            torch.save({'state_dict': model.state_dict()}, './models/' + args.model_id + '.pth')
            print(f'checkpoint saved at epoch {epoch}')
            best = curr_gap

        # print epoch_stats
        if epoch % args.resultPrintFreq == 0 or epoch == args.epochs - 1:
            print('----- Epoch {} -----'.format(epoch))
            print('Train Loss: {:.5f}, '
                  'Train Time: {: .5f}'.format(curr_loss,
                                               epoch_stats['train_time']))
            print('Val Gap: {:.5f}, '
                  'Val Gap Worst: {: .5f}, '
                  'Val Iters: {: d}, '
                  'Val Time: {: .5f}'.format(curr_gap,
                                             epoch_stats['val_gap_worst'],
                                             int(epoch_stats['val_iters'] // epoch_stats['val_agg']),
                                             epoch_stats['val_time'] / epoch_stats['val_agg']))

            print('Equality Violation: {:.5f}, '
                  'Equality Violation Worst: {: .5f}, '
                  'Inequality Violation: {:.5f}, '
                  'Inequality Violation Worst: {: .5f}, '.format(epoch_stats['val_eq'] / epoch_stats['val_agg'],
                                                                 epoch_stats['val_eq_worst'],
                                                                 epoch_stats['val_ineq'] / epoch_stats['val_agg'],
                                                                 epoch_stats['val_ineq_worst'],
                                                                 ))

        if epoch % args.resultSaveFreq == 0 or epoch == args.epochs - 1:
            stats = epoch_stats
            with open(f'./logs/run_training/{args.model_id}_TrainingStats.dict', 'wb') as f:
                pickle.dump(stats, f)


def featurize_batch(args, batch):
    with torch.no_grad():
        bsz = batch.feature.shape[0]
        A_sp = torch.sparse_coo_tensor(indices=batch.A_indices,
                                        values=batch.A_values,
                                        size=(bsz * args.constr_num, bsz * args.var_num)).to(args.device)
        batch = batch.to(args.device)
        return batch, A_sp


def optimizer_step(model, feasibility_net, optimizer, batch, args, problem, epoch_stats):
    #torch.cuda.reset_peak_memory_stats(args.device)

    start_time = time.time()
    optimizer.zero_grad()
    train_loss = get_loss(model, feasibility_net, batch, problem, args, args.loss_type)

    # print(f"Step {args.epoch}: After forward pass of loss: {torch.cuda.memory_allocated(args.device) / 1e6:.2f} MB")

    train_loss.backward()

    #print(f"Step {args.epoch}: After backwartd pass: {torch.cuda.memory_allocated(args.device) / 1e6:.2f} MB")

    optimizer.step()

    #print(f"Step {args.epoch}: After optimizer step: {torch.cuda.memory_allocated(args.device) / 1e6:.2f} MB")

    train_time = time.time() - start_time
    dict_agg(epoch_stats, 'train_time', train_time)
    dict_agg(epoch_stats, 'train_loss', float(train_loss.detach().cpu()))
    dict_agg(epoch_stats, 'train_agg', 1.)


def get_loss(model, feasibility_net, batch, problem, args, loss_type):
    batch, A_sp = featurize_batch(args, batch)

    if args.problem == 'primal_lp':
        x = model(batch.feature)

        G = batch.G if hasattr(batch, 'G') else None
        A_eq = batch.A_eq if hasattr(batch, 'A_eq') else None

        x_feas = feasibility_net(x=x, A=A_sp, b=batch.b,
                                 nonnegative_mask=problem.nonnegative_mask, feature=batch.feature,
                                 G=G, A_eq=A_eq)
    else:
        raise ValueError('Invalid problem')

    predicted_obj = problem.obj_fn(x=x_feas)

    if loss_type == 'obj':

        if args.algo in ['DC3', 'POCS', 'DC3LHS']:
            eq_residual = problem.eq_residual(x_feas, A_sp, batch.b)
            ineq_residual = problem.ineq_residual(x_feas, G)
            eq_violation = torch.norm(eq_residual, p=2, dim=-1)
            ineq_violation = torch.norm(ineq_residual, p=2, dim=-1)
            return (predicted_obj +
                    args.dc3_softweight * (1 - args.dc3_softweighteqfrac) * ineq_violation +
                    args.dc3_softweight * args.dc3_softweighteqfrac * eq_violation).mean()

        return predicted_obj.mean()

    elif loss_type == 'gap':
        return problem.optimality_gap(predicted_obj, batch.target).mean()

    elif loss_type == 'mse':
        assert args.algo == 'LDRPM' or 'LDRPMLHS'
        return F.mse_loss(x, feasibility_net.algo.x_LDR)

    else:
        raise ValueError('Invalid loss_type')


def validate_model(model, feasibility_net, args, data, problem, epoch_stats):
    for i, batch in enumerate(data['val']):
        start_time = time.time()
        optimality_gap = get_loss(model, feasibility_net, batch, problem, args, "gap")
        val_time = time.time() - start_time

        gap = float(optimality_gap.mean().detach().cpu())
        gap_worst = float(torch.norm(optimality_gap, float('inf')).detach().cpu())  # max of samples
        dict_agg(epoch_stats, 'val_time', val_time)
        dict_agg(epoch_stats, 'val_gap', gap)
        dict_agg(epoch_stats, 'val_gap_worst', gap_worst)
        dict_agg(epoch_stats, 'val_agg', 1.)

        iters = feasibility_net.iters
        eq_epsilon = float(feasibility_net.eq_epsilon.mean().detach().cpu())
        ineq_epsilon = float(feasibility_net.ineq_epsilon.mean().detach().cpu())
        eq_epsilon_worst = float(torch.norm(feasibility_net.eq_epsilon, float('inf')).detach().cpu())
        ineq_epsilon_worst = float(torch.norm(feasibility_net.ineq_epsilon, float('inf')).detach().cpu())
        dict_agg(epoch_stats, 'val_iters', iters)
        dict_agg(epoch_stats, 'val_eq', eq_epsilon)
        dict_agg(epoch_stats, 'val_ineq', ineq_epsilon)
        dict_agg(epoch_stats, 'val_eq_worst', eq_epsilon_worst)
        dict_agg(epoch_stats, 'val_ineq_worst', ineq_epsilon_worst)


def dict_agg(stats, key, value):
    if key in stats.keys():
        if "worst" in key:
            stats[key] = max(stats[key], value)
        else:
            stats[key] += value
    else:
        stats[key] = value


def evaluate_model(args, data, problem):
    test_stats = {}
    test_gaps = []
    model = load_model(args)
    model = load_weights(model, args)
    feasibility_net = load_algo(args)
    model.eval()
    feasibility_net.eval()
    for i, batch in enumerate(data['test']):
        start_time = time.time()
        optimality_gap = get_loss(model, feasibility_net, batch, problem, args, "gap")
        test_time = time.time() - start_time

        gap = float(optimality_gap.mean().detach().cpu())
        gap_worst = float(torch.norm(optimality_gap, float('inf')).detach().cpu())  # max of samples
        dict_agg(test_stats, 'test_time', test_time)
        dict_agg(test_stats, 'test_gap', gap)
        dict_agg(test_stats, 'test_gap_worst', gap_worst)
        dict_agg(test_stats, 'test_agg', 1.)
        test_gaps.append(gap)

        iters = feasibility_net.iters
        eq_epsilon = float(feasibility_net.eq_epsilon.mean().detach().cpu())
        ineq_epsilon = float(feasibility_net.ineq_epsilon.mean().detach().cpu())
        eq_epsilon_worst = float(torch.norm(feasibility_net.eq_epsilon, float('inf')).detach().cpu())
        ineq_epsilon_worst = float(torch.norm(feasibility_net.ineq_epsilon, float('inf')).detach().cpu())
        dict_agg(test_stats, 'test_iters', iters)
        dict_agg(test_stats, 'test_iters_worst', iters)
        dict_agg(test_stats, 'test_eq', eq_epsilon)
        dict_agg(test_stats, 'test_ineq', ineq_epsilon)
        dict_agg(test_stats, 'test_eq_worst', eq_epsilon_worst)
        dict_agg(test_stats, 'test_ineq_worst', ineq_epsilon_worst)

    test_stats['test_gaps'] = np.array(test_gaps)

    with open(f'./logs/run_training/{args.model_id}_TestStats.dict', 'wb') as f:
        pickle.dump(test_stats, f)

    calculate_scores(args, data)


def calculate_scores(args, data):
    if os.path.exists(f'./logs/run_training/{args.model_id}_TrainingStats.dict'):
        try:
            with open(f'./logs/run_training/{args.model_id}_TrainingStats.dict', 'rb') as f:
                training_stats = pickle.load(f)
        except:
            print(f'{args.model_id}_TrainingStats.dict is missing. Load test stats only.')

    with open(f'./logs/run_training/{args.model_id}_TestStats.dict', 'rb') as f:
        test_stats = pickle.load(f)

    # store the test gap
    np.save(f'./data/results_summary/{args.model_id}_test_gaps.npy', test_stats['test_gaps'])

    scores = {'test_optimality_gap_mean': test_stats['test_gap'] / test_stats['test_agg'],
              'test_optimality_gap_worst': test_stats['test_gap_worst'],
              'test_eq_violation_mean': test_stats['test_eq'] / test_stats['test_agg'],
              'test_eq_violation_worst': test_stats['test_eq_worst'],
              'test_ineq_violation_mean': test_stats['test_ineq'] / test_stats['test_agg'],
              'test_ineq_violation_worst': test_stats['test_ineq_worst'],
              'test_iters_mean': int(test_stats['test_iters'] // test_stats['test_agg']),
              'test_iters_worst': int(test_stats['test_iters_worst']),
              'train_time': training_stats['train_time'],
              'val_time': training_stats['val_time'] / training_stats['val_agg'],
              'test_time': test_stats['test_time'] / test_stats['test_agg'],}
    print(scores)
    create_report(scores, args)


def create_report(scores, args):
    args_dict = args_to_dict(args)
    # combine scores and args dict
    args_scores_dict = args_dict | scores
    # save dict
    save_dict(args_scores_dict, args)
    plot_distribution(args)


def args_to_dict(args):
    return vars(args)


def save_dict(dictionary, args):
    w = csv.writer(open('./data/results_summary/' + args.model_id + '.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])


def plot_distribution(args):
    test_gaps = np.load(f'./data/results_summary/{args.model_id}_test_gaps.npy')

    # Plot and save test_gap histogram
    plt.figure()
    plt.hist(test_gaps, bins=100)
    plt.xlabel('test gap')
    plt.savefig(f'./data/results_summary/{args.model_id}_test_gap_hist.pdf', format='pdf')


def load_weights(model, args):
    PATH = './models/' + args.model_id + '.pth'
    #checkpoint = torch.load(PATH, map_location=args.device, weights_only=False)
    checkpoint = torch.load(PATH, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    return model
