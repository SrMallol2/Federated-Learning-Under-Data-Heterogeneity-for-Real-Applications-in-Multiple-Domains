from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
# Implementation for FedAvg Server
import time


class FedAvg(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all  users
        data = read_data(args.dataset, args.dataset_path)
        total_users = len(data[0])
        self.use_adam = 'adam' in self.algorithm.lower()
        print("Users in total: {}".format(total_users))

        for i in range(total_users):
            id, train_data, test_data = read_user_data(i, data, dataset=args.dataset, model=args.model)
            user = UserAVG(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("Number of users / total users:", args.num_users, " / ", total_users)
        print("Finished creating FedAvg server.")

    def train(self, args):
        best_evaluation_metric = None
        best_evaluation_metric_epoch = None
        current_evaluation_metric = None
        last_improvement = 0
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            self.send_parameters(mode=self.mode)
            self.evaluate()
            current_evaluation_metric = self.metrics['glob_test_metric'][-1][args.early_stopping_criteria]
            if best_evaluation_metric is None or current_evaluation_metric < best_evaluation_metric:
                self.save_model()
                for user in self.selected_users:
                    user.save_model()
                print('The model is saved!')
                best_evaluation_metric = current_evaluation_metric
                best_evaluation_metric_epoch = glob_iter
                last_improvement = 0
            else:
                print(
                    'Best Validation Evaluation Metric: {}, Best Epoch: {}'.format(
                        best_evaluation_metric, best_evaluation_metric_epoch
                    ))
                last_improvement += 1
            if args.early_stopping == 'True' and last_improvement > args.early_stopping_patience:
                print('No improvement was observed after {} epochs.'.format(last_improvement))
                print(
                    'The best model with the evaluation metric of {} on the validation set was saved at epoch {}.'
                    .format(best_evaluation_metric, best_evaluation_metric_epoch))
                break
            self.timestamp = time.time()  # log user-training start time
            for user in self.selected_users:  # allow selected users to train
                user.train(glob_iter, personalized=self.personalized)  # * user.train_samples
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            # Evaluate selected user
            if self.personalized:
                # Evaluate personal model on user for each iteration
                print("Evaluate personal model\n")
                self.evaluate_personalized_model()

            self.timestamp = time.time()  # log server-agg start time
            self.aggregate_parameters(partial=(self.mode != 'all'))
            curr_timestamp = time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
        self.save_results(args)

