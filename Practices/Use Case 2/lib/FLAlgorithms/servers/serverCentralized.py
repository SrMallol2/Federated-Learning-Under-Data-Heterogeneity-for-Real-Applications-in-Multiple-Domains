from FLAlgorithms.users.userCentralized import UserCentralized
from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
# Implementation for FedAvg Server
import time


class Centralized(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all  users
        data = read_data(args.dataset, args.dataset_path)
        total_users = len(data[0])
        self.use_adam = True
        print("Users in total: {}".format(total_users))

        aggregated_train_data = []
        aggregated_test_data = []
        for i in range(total_users):
            id, train_data, test_data = read_user_data(i, data, dataset=args.dataset, model=args.model)
            aggregated_test_data.extend(test_data)
            aggregated_train_data.extend(train_data)

        user = UserCentralized(args, 0, model, aggregated_train_data, aggregated_test_data, use_adam=True)
        self.users.append(user)
        self.total_train_samples += user.train_samples

        print("Number of users / total users:", args.num_users, " / ", total_users)
        print("Finished creating Centralized server.")

    def train(self, args):
        best_evaluation_metric = None
        best_evaluation_metric_epoch = None
        current_evaluation_metric = None
        last_improvement = 0
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter, self.num_users)
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
        self.save_results(args)
