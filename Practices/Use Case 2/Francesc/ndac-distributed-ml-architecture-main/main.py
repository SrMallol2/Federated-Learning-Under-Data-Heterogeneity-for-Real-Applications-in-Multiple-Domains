#!/usr/bin/env python
import argparse

import eco2ai
import torch

from FLAlgorithms.servers.serverCentralized import Centralized
from FLAlgorithms.servers.serverFedDistill import FedDistill
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverIsolated import Isolated
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedEnsemble import FedEnsemble
from FLAlgorithms.servers.serverpFedGen import FedGen
from FLAlgorithms.servers.serverpFedGenSpecified import FedGenSpecified
from utils.model_utils import create_model


def create_server_n_user(args, i):
    model = create_model(args.model, args.dataset, args.algorithm, args.problem_type, args.output_range, args.steps)
    model[0].to(args.device)
    if ('FedAvg' in args.algorithm):
        server = FedAvg(args, model, i)
    elif 'FedGen' in args.algorithm:
        if args.specified_mode:
            server = FedGenSpecified(args, model, i)
        else:
            server = FedGen(args, model, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, i)
    elif ('FedDistill' in args.algorithm):
        server = FedDistill(args, model, i)
    elif ('FedEnsemble' in args.algorithm):
        server = FedEnsemble(args, model, i)
    elif ('Centralized' in args.algorithm):
        server = Centralized(args, model, i)
    elif ('Isolated' in args.algorithm):
        server = Isolated(args, model, i)
    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        if args.track_energy_consumption:
            args.tracker.start()
        server.train(args)
        if args.track_energy_consumption:
            args.tracker.stop()
    server.test()


def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FlagsRegression-user20-alpha0.0001-ratio1")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/dsalami/projects/FedGen/data/FlagsRegression/ApproachComparison/lookback_60/steps_1")
    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--algorithm", type=str, default="FedGen")
    parser.add_argument("--problem_type", type=str, default="regression")
    parser.add_argument("--output_range", type=list, default=[1, 2], help='only used for the regression problem')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01,
                        help="Personalized learning rate to calculate theta approximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=1, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=50)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cuda:1", help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str,
                        default="/home/dsalami/projects/FedGen/data/FlagsRegression/EnergyConsumption/lookback_60/steps_1/FedGen/lstm/rep_0",
                        help="directory path to save results")
    parser.add_argument("--specified_mode", type=bool, default=False,
                        help="in specified mode special APs are selected for training. Check the dataset generation script.")
    parser.add_argument('--lookback', type=int, default=60,
                        help='Number of past samples in the time series [default: 5]')
    parser.add_argument('--steps', type=int, default=1, help='Number of future samples in time series [default: 1]')
    parser.add_argument('--track_energy_consumption', type=bool, default=True,
                        help='Track the energy consumption [default: False]')
    parser.add_argument('--early_stopping_criteria', default='unscaled_mae',
                        help='What criteria to use to stop the training. Model will be saved if the metric is LOWER than the best. [default: unscaled_mae]')
    parser.add_argument('--early_stopping', default='True', help='Whether to use early stopping [default: True]')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Stop the training if there is no improvements after this ' +
                             'number of consequent epochs [default: 50]')

    args = parser.parse_args()

    if args.track_energy_consumption:
        args.tracker = eco2ai.Tracker(
            project_name=f"{args.algorithm}/{args.model}/{args.lookback}/{args.steps}",
            experiment_description=f"Training {args.algorithm}/{args.model}",
            file_name=f"{args.result_path}/emission.csv",
            alpha_2_code='FI'
        )

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
