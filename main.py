from train import run_training
from utils import load_data, load_problem, get_ldr_result
import argparse
import os
import torch
import yaml
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_idx", help="config index", type=int, default=0)
    parser.add_argument("--dataset", default="portfolio_optimization")
    parser.add_argument("--problem", help="primal_lp", default="primal_lp")
    parser.add_argument("--job", default="training", type=str)

    # save related parameters
    parser.add_argument("--resultSaveFreq", default=1000, type=int)
    parser.add_argument("--resultPrintFreq", default=2, type=int)
    parser.add_argument("--float64", default=True, type=bool)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser.add_argument("--continue_training", default=False, type=str2bool)
    return parser.parse_args()


def complete_args(cfg_file, problem_json, init_args):
    init_args_dict = vars(init_args)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    with open(problem_json, "r") as f:
        problem_info = json.load(f)

    args_dict = {**init_args_dict, **cfg, **problem_info}
    args = argparse.Namespace(**args_dict)

    # Override
    if args.epochs < args.resultSaveFreq:
        args.resultSaveFreq = args.epochs
    args.model_id = f"{args.dataset}_cfg{args.cfg_idx}"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.feature_dim = args.feature_num
    if args.model == 'mlp':
        args.out_dim = args.var_num
    if args.algo == 'DC3':
        if args.changing_feature == 'b':
            args.out_dim = args.var_num - args.constr_num
        if args.changing_feature == 'A':
            args.out_dim = args.var_num - args.x_nneg_num - args.eq_constr_num
            args.algo = 'DC3LHS'
    if args.algo == 'LDRPM':
        if args.changing_feature == 'A':
            args.algo = 'LDRPMLHS'
    if args.algo == 'POCS':
        if args.changing_feature == 'A':
            args.algo = 'POCSLHS'

    # Assert

    return args


def main(args):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./data/prediction'):
        os.makedirs('./data/prediction')
    if not os.path.exists('./data/results_summary'):
        os.makedirs('./data/results_summary')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./logs/run_training'):
        os.makedirs('./logs/run_training')

    if args.float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if args.job == 'training':
        problem = load_problem(args)
        data = load_data(args)
        run_training(args, data, problem)
    elif args.job == 'get_ldr_result':
        get_ldr_result(args)
    else:
        raise ValueError('Invalid job type')


if __name__ == '__main__':
    init_args = add_arguments()
    cfg_file = f"./cfg/{init_args.dataset}_{init_args.cfg_idx}"
    metadata_json = f"./data/{init_args.dataset}/new_feasibility/metadata.json"
    args = complete_args(cfg_file, metadata_json, init_args)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    main(args)