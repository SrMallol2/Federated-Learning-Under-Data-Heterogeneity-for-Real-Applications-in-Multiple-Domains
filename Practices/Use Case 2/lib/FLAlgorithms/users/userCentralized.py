import torch
from tqdm import tqdm

from FLAlgorithms.users.userbase import User


class UserCentralized(User):
    def __init__(self, args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

    def train(self, glob_iter, personalized=False, lr_decay=True):
        self.model.train()
        for _ in tqdm(range(len(self.trainloader))):
            result = self.get_next_train_batch(return_y_distribution=False)
            X, y = result['X'], result['y']
            X = X.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)['output']
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()  # self.plot_Celeb)

            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
            # local-model ===> self.model
            # self.clone_model_paramenter(self.local_model, self.model.parameters())
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
