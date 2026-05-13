import copy
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader

from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from utils.dataset_preprocessings import INVERSE_PREPROCESSING
from utils.model_config import RUNCONFIGS
from utils.model_utils import get_dataset_name


class User:
    """
    Base class for users in federated learning.
    """

    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False):
        self.device = args.device
        self.model = copy.deepcopy(model[0])
        self.model = self.model.to(self.device)
        self.model_name = model[1]
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.problem_type = args.problem_type
        self.save_path = args.result_path
        # self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False)
        self.trainloader = DataLoader(train_data,
                                      self.batch_size if len(train_data) >= self.batch_size else len(train_data),
                                      shuffle=True, drop_last=True)
        self.testloader = DataLoader(test_data, self.batch_size * 100, drop_last=False)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        dataset_name = get_dataset_name(self.dataset)
        self.inverse_preprocessing = INVERSE_PREPROCESSING[dataset_name]
        dataset_ = self.dataset.lower().replace('user', '').replace('alpha', '').replace('ratio', '').split('-')
        user, alpha, ratio = dataset_[1], dataset_[2], dataset_[3]
        self.dataset_path_prefix = os.path.join(args.dataset_path,
                                                'u{}-alpha{}-ratio{}'.format(user, alpha, ratio))
        self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()
        if use_adam:
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}
        self.y_distribution = {
            'mean': None,
            'std': None,
            'n': 0
        }

    def weighted_mse_loss(self, output, target):
        return torch.sum((target ** 2) * (output - target) ** 2)

    def init_loss_fn(self):
        if self.problem_type == 'classification':
            self.loss = nn.NLLLoss()
        elif self.problem_type == 'regression':
            self.loss = nn.L1Loss()
        self.dist_loss = nn.MSELoss()
        if self.problem_type == 'classification':
            self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        elif self.problem_type == 'regression':
            self.ensemble_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()

    def set_parameters(self, model, beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()

    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()

    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params, keyword='all'):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_metrics = {}
        loss = 0
        for x, y in self.testloader:
            with torch.no_grad():
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)['output']
                loss += self.loss(output, y)
                output_numpy = output.cpu().detach().numpy()
                y_numpy = y.cpu().numpy()
                if self.problem_type == 'classification':
                    if 'accuracy' not in test_metrics:
                        test_metrics['accuracy'] = 0
                    test_metrics['accuracy'] += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                elif self.problem_type == 'regression':
                    if 'mse' not in test_metrics:
                        test_metrics['mse'] = 0
                    if 'mae' not in test_metrics:
                        test_metrics['mae'] = 0
                    if 'mape' not in test_metrics:
                        test_metrics['mape'] = 0
                    if 'unscaled_mse' not in test_metrics:
                        test_metrics['unscaled_mse'] = 0
                    if 'unscaled_mae' not in test_metrics:
                        test_metrics['unscaled_mae'] = 0
                    if 'unscaled_mape' not in test_metrics:
                        test_metrics['unscaled_mape'] = 0
                    unscaled_output = self.inverse_preprocessing(
                        self.dataset_path_prefix, 'train',
                        np.repeat([output_numpy.squeeze()], x.shape[-1]).reshape(-1, x.shape[-1]), axis=0)
                    unscaled_y = self.inverse_preprocessing(
                        self.dataset_path_prefix, 'train',
                        np.repeat([y_numpy.squeeze()], x.shape[-1]).reshape(-1, x.shape[-1]), axis=0)
                    test_metrics['mse'] += mean_squared_error(y_numpy, output_numpy)
                    test_metrics['mae'] += mean_absolute_error(y_numpy, output_numpy)
                    test_metrics['mape'] += mean_absolute_percentage_error(y_numpy, output_numpy)

                    test_metrics['unscaled_mse'] += mean_squared_error(unscaled_y, unscaled_output)
                    test_metrics['unscaled_mae'] += mean_absolute_error(unscaled_y, unscaled_output)
                    test_metrics['unscaled_mape'] += mean_absolute_percentage_error(unscaled_y, unscaled_output)
        if self.problem_type == 'classification':
            test_metrics['accuracy'] /= len(self.testloader)
        elif self.problem_type == 'regression':
            test_metrics['mse'] /= len(self.testloader)
            test_metrics['mae'] /= len(self.testloader)
            test_metrics['mape'] /= len(self.testloader)
            test_metrics['unscaled_mse'] /= len(self.testloader)
            test_metrics['unscaled_mae'] /= len(self.testloader)
            test_metrics['unscaled_mape'] /= len(self.testloader)
        loss /= len(self.testloader)
        return test_metrics, loss, y.shape[0]

    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            output = self.model(x)['output']
            loss += self.loss(output, y)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            # @loss += self.loss(output, y)
            # print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            # print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0], loss

    def get_next_train_batch(self, return_y_distribution=True):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        if return_y_distribution:
            if self.problem_type == 'classification':
                unique_y, counts = torch.unique(y, return_counts=True)
                unique_y = unique_y.detach().numpy()
                counts = counts.detach().numpy()
                result['labels'] = unique_y
                result['counts'] = counts
            elif self.problem_type == 'regression':
                mean, std = norm.fit(y)
                result['mean'] = mean
                result['std'] = std
                result['n'] = len(y)
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join(self.save_path, "models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + str(self.id) + ".pt"))

    def load_model(self):
        model_path = os.path.join(self.save_path, "models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "user_" + str(self.id) + ".pt"))
        self.model.to(self.device)
