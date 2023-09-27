import os, json, warnings
from pathlib import Path
from datetime import datetime
import copy

import numpy as np
import pandas as pd
import pickle

import torch
from torch.nn import *

from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
# from sklearn.externals import joblib

# Sklearn Models
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Metrics
from torcheval.metrics.functional import multiclass_auprc, multiclass_confusion_matrix, binary_accuracy
from sklearn.metrics import log_loss, confusion_matrix, balanced_accuracy_score, f1_score

from sklearn.exceptions import ConvergenceWarning
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
    # Edited Files: summary.py [cumsum @ 386]
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from dataset import Loader

warnings.simplefilter("ignore", category=ConvergenceWarning)

class Classifier:
    def __init__(self, loader, model=None, optimizer=None, loss_fn=None, scheduler=None,
                 model_type='torch', model_name=None, model_params=None,
                 seed=1234, output_dir='outputs', resume=None, mode='train',
                 lr=0.001):
        self.mode = mode
        self.seed = seed
        torch.manual_seed(self.seed)

        self.model_type = model_type
        self.model_name = model_name
        self.model_params = model_params
        self.predict_prob = True

        self.output_dir, self.log_dir, self.model_dir, self.report_dir, self.report_model_dir = self.load_output_dirs(output_dir)

        self.log_name = datetime.now().strftime("%d-%m-%Y %H_%M")
        if not loader.cross_valid:
            self.model_path = os.path.join(self.model_dir, self.log_name + '000')
        else:
            self.model_paths = [os.path.join(self.model_dir, self.log_name + f'_k{str(k)}' + '000') for k in range(loader.kfolds)]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.loader = loader

        if not self.loader.cross_valid:
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            if self.loss_fn is None:
                self.loss_fn = log_loss
        else:
            self.models = [copy.deepcopy(model) for _ in range(self.loader.kfolds)]
            self.loss_fns = [CrossEntropyLoss() for _ in range(self.loader.kfolds)]
            self.optimizers = [torch.optim.Adam(self.models[i].parameters(), lr=lr) for i in range(self.loader.kfolds)]

        if self.mode == 'train':
            self.resume(resume)
            if not self.loader.cross_valid:
                if not self.model is None and self.model_type == 'torch':
                    # Send model to device
                    self.model.to(self.device)
                    self.scheduler = scheduler
            else:
                for k in range(self.loader.kfolds):
                    self.models[k].to(self.device)
                    self.scheduler = scheduler

            if self.model_type != 'torch' and self.model is None and self.model_params:
                self.model = self.get_scikit_model(self.model_params)
        else:
            if self.model_type == 'torch':
                self.model.to(self.device)

        if not self.loader.cross_valid:
            self.board_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.log_name),
                                              comment=self.log_name[:-3],
                                              filename_suffix=self.log_name[:-3])
        else:
            self.board_writers = [SummaryWriter(log_dir=os.path.join(self.log_dir, f'{self.log_name}_{k}'), comment=f'{self.log_name[:-3]}_{k}',
                                              filename_suffix=f'{self.log_name[:-3]}_{k}')
                                  for k in range(self.loader.kfolds)]

        self.available_models = ('gaussian_nb', 'multi_nb', 'bern_nb', 'glm', 'knn', 'svm', 'r_forest', 'lda', 'qda')
        # self.board_writer.add_graph(self.model, torch.zeros(next(self.model.parameters()).size()).to(self.device))

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        self.gain_matrix = torch.tensor([[0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
                                        [-0.25, 1, -0.3, -0.1, -0.1, -0.1, -0.1],
                                        [-0.02, -0.1, 1, -0.1, -0.1, -0.1, -0.1],
                                        [-0.25, -0.1, -0.3, 1, -0.1, -0.1, -0.1],
                                        [-0.25, -0.1, -0.3, -0.1, 1, -0.1, -0.1],
                                        [-0.25, -0.1, -0.3, -0.1, -0.1, 1, -0.1],
                                        [-0.25, -0.1, -0.3, -0.1, -0.1, -0.1, 1]]).to(self.device)

    def load_output_dirs(self, output_dir):
        self.output_dir = output_dir
        self.log_dir = os.path.join(self.output_dir, 'log')
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.report_dir = os.path.join(self.output_dir, 'reports')
        self.report_model_dir = os.path.join(self.report_dir, self.model_name)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_model_dir).mkdir(parents=True, exist_ok=True)
        return self.output_dir, self.log_dir, self.model_dir, self.report_dir, self.report_model_dir

    @staticmethod
    def normalize(x):
        return (x - x.mean()) / x.std()

    def average_precision(self, y_pred, y, average='macro'):
        if not self.loader.binary:
            return balanced_accuracy_score(torch.argmax(y, dim=1).squeeze().cpu().numpy(), torch.argmax(y_pred, dim=1).squeeze().cpu().numpy())
            # return multiclass_auprc(y_pred, torch.argmax(y, dim=1), average=average)
        else:
            return balanced_accuracy_score(torch.argmax(y, dim=1).squeeze().cpu().numpy(), torch.argmax(y_pred, dim=1).squeeze().cpu().numpy())
            # return binary_accuracy(torch.argmax(y_pred, dim=1).squeeze(), torch.argmax(y, dim=1).squeeze())

    def confusion_matrix(self, y, y_preds):
        if self.model_type == 'torch':
            if self.loader.binary:
                return multiclass_confusion_matrix(y_preds, torch.argmax(y, dim=1), 2)
            elif self.loader.non_binary:
                return multiclass_confusion_matrix(y_preds, torch.argmax(y, dim=1), 6)
            else:
                return multiclass_confusion_matrix(y_preds, torch.argmax(y, dim=1), 7)
        else:
            try:
                conf_mat = confusion_matrix(y, np.argmax(y_preds, axis=1))
            except:
                conf_mat = confusion_matrix(y, y_preds)
            return pd.DataFrame(conf_mat)

    def total_savings(self, y, y_preds):
        return (self.gain_matrix * self.confusion_matrix(y, y_preds)).sum()

    def predict(self, x, y):
        if self.model_type == 'torch':
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
        else:
            y_pred = self.model.predict_proba(x)
        return x, y, y_pred

    def train(self, epochs=0, param=None):
        # TODO Add Manual Plots
        i = 0
        if self.model_type == 'torch':
            torch.manual_seed(self.seed)
            if not self.loader.cross_valid:
                self.train_torch_fold(self.model, self.loss_fn, self.optimizer,
                                      self.loader.train, self.loader.valid,
                                      epochs)
                # self.model.train()
                # i, train_loss, train_prec, train_total_save,\
                #     val_loss, val_prec, valid_total_save,\
                #     test_loss, test_prec, test_total_save\
                #     = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                # for i in range(init_epochs, init_epochs + epochs):
                #     train_loss, train_acc, train_prec, train_total_save, train_conf_mat = self.train_torch_epoch(i)
                #     val_loss, val_acc, val_prec, valid_total_save, valid_conf_mat = self.validate_torch_epoch(self.loader.valid)
                #     self.board_update(i, train_loss, train_prec, train_total_save, val_loss, val_prec, valid_total_save, train_conf_mat, valid_conf_mat)
                #     self.save_checkpoint(i)
                #     train_losses.append(train_loss)
                #     valid_losses.append(val_loss)
                #     train_avg_precs.append(train_prec)
                #     valid_avg_precs.append(val_prec)
                #     train_total_saves.append(train_total_save)
                #     valid_total_saves.append(valid_total_save)
                # test_loss, test_acc, test_prec, test_total_save, test_conf_mat = self.validate_torch_epoch(self.loader.test)
                # results = pd.DataFrame({'Train/Loss': train_losses, 'Valid/Loss': valid_losses,
                #                         'Train/Precision': train_avg_precs, 'Valid/Precision': valid_avg_precs})
                # self.board_update_hp(i, train_loss, train_prec, train_total_save, val_loss, val_prec, valid_total_save, test_loss, test_prec, test_total_save)
                # self.save_df(results, prefix=self.model_name)
            else:
                for k in range(self.loader.kfolds):
                    self.train_torch_fold(self.models[k], self.loss_fns[k], self.optimizers[k],
                                          self.loader.train_folds[k], self.loader.valid_folds[k],
                                          epochs, k=k)
        else:
            if not param is None:
                train_loss, val_loss, test_loss, train_prec, val_prec, test_prec = self.train_scikit_model(param)
                self.save_checkpoint(i)
            else:
                raise AttributeError('Parameters for sklearn model missing')
        # self.board_update_hp(i, train_loss, train_prec, train_total_save, val_loss, val_prec, valid_total_save, test_loss, test_prec, test_total_save)

    def explore(self, hyperparameters=None, param_name=''):
        if self.model_type == 'torch':
            raise NotImplementedError('PyTorch Exploration Not yet implemented')
        else:
            return self.explore_scikit(hyperparameters, param_name=param_name)

    def train_torch_fold(self, model, loss_fn, optimizer, train_ds, valid_ds, epochs=0, k=0):
        train_losses, valid_losses, test_losses = [], [], []
        train_avg_precs, valid_avg_precs, test_avg_precs = [], [], []
        train_total_saves, valid_total_saves, test_total_saves = [], [], []
        if not self.loader.cross_valid:
            init_epochs = int(self.model_path[-3:]) + 1 if int(self.model_path[-3:]) != 0 else 0
        else:
            init_epochs = int(self.model_paths[k][-3:]) + 1 if int(self.model_paths[k][-3:]) != 0 else 0
        model.train()
        i, train_loss, train_prec, train_total_save, \
            val_loss, val_prec, valid_total_save, \
            test_loss, test_prec, test_total_save \
            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        for i in range(init_epochs, init_epochs + epochs):
            train_loss, train_acc, train_prec, train_total_save, train_conf_mat =\
                self.train_torch_epoch(i, model, loss_fn, optimizer, train_ds, k)
            val_loss, val_acc, val_prec, valid_total_save, valid_conf_mat =\
                self.validate_torch_epoch(model, loss_fn, valid_ds)
            self.board_update(i, train_loss, train_prec, train_total_save, val_loss, val_prec, valid_total_save,
                              train_conf_mat, valid_conf_mat, k)
            self.save_checkpoint(i, model, optimizer, k)
            train_losses.append(train_loss)
            valid_losses.append(val_loss)
            train_avg_precs.append(train_prec)
            valid_avg_precs.append(val_prec)
            train_total_saves.append(train_total_save)
            valid_total_saves.append(valid_total_save)
        results = pd.DataFrame({'Train/Loss': train_losses, 'Valid/Loss': valid_losses,
                                'Train/Precision': train_avg_precs, 'Valid/Precision': valid_avg_precs})
        if not self.loader.cross_valid:
            test_loss, test_acc, test_prec, test_total_save, test_conf_mat = self.validate_torch_epoch(model, loss_fn, self.loader.test)
            self.board_update_hp(i, train_loss, train_prec, train_total_save, val_loss, val_prec, valid_total_save,
                                 test_loss, test_prec, test_total_save, k)
        self.save_df(results, prefix=f'{self.model_name}_k{str(k)}')
        return results

    def train_torch_epoch(self, epoch, model, loss_fn, optimizer, train_ds, k):
        total, correct, sum_loss = 0, 0, 0
        i, prec, total_savings = 0, 0, 0
        if self.loader.binary:
            ys, y_preds = torch.empty(size=(0, 2)).to(self.device), torch.empty(size=(0, 2)).to(self.device)
        elif self.loader.non_binary:
            ys, y_preds = torch.empty(size=(0, 6)).to(self.device), torch.empty(size=(0, 6)).to(self.device)
        else:
            ys, y_preds = torch.empty(size=(0,7)).to(self.device), torch.empty(size=(0,7)).to(self.device)
        with tqdm(train_ds) as pbar:
            pbar.set_description(f'K{k} - EP {epoch}')
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x)
                # print(y_pred)
                optimizer.zero_grad()

                loss = loss_fn(y_pred, y)
                loss.backward()
                # utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                sum_loss += loss.cpu().item() * y.shape[0]
                correct += (y_pred == y).cpu().float().sum()
                total += y.shape[0]
                prec += self.average_precision(y_pred.cpu(), y.cpu())

                ys = torch.cat([ys, y])
                y_preds = torch.cat([y_preds, y_pred])

                i += 1
            if self.scheduler:
                if isinstance(self.scheduler, float):
                    if prec / (i + 1) >= self.scheduler:
                        for g in self.optimizer.param_groups:
                            g['lr'] *= 0.5
                    self.scheduler *= 1.05
                    # self.optim.param_groups[0]['lr'] *= 0.1
                else:
                    self.scheduler.step()
            # print(len(ys), len(y_preds))
            # ys = torch.tensor(ys)
            # y_preds = torch.tensor(y_preds)
            conf_mat = self.confusion_matrix(ys, y_preds)
            if not self.loader.binary and not self.loader.non_binary:
                total_savings = self.total_savings(ys, y_preds)
        return sum_loss/total, correct/total, prec/i, total_savings, conf_mat

    def validate_torch_epoch(self, model, loss_fn, valid_ds):
        sum_loss, correct, total = 0, 0, 0
        j, prec, total_savings = 0, 0, 0
        if self.loader.binary:
            ys, y_preds = torch.empty(size=(0, 2)).to(self.device), torch.empty(size=(0, 2)).to(self.device)
        elif self.loader.non_binary:
            ys, y_preds = torch.empty(size=(0, 6)).to(self.device), torch.empty(size=(0, 6)).to(self.device)
        else:
            ys, y_preds = torch.empty(size=(0,7)).to(self.device), torch.empty(size=(0,7)).to(self.device)
        with torch.no_grad():
            for x, y in tqdm(valid_ds):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x)

                loss = loss_fn(y_pred, y)
                sum_loss += loss.cpu().item() * y.shape[0]
                correct += (y_pred == y).cpu().float().sum()
                total += y.shape[0]
                prec += self.average_precision(y_pred.cpu(), y.cpu())
                j += 1

                ys = torch.cat([ys, y])
                y_preds = torch.cat([y_preds, y_pred])

            conf_mat = self.confusion_matrix(ys, y_preds)
            if not self.loader.binary and not self.loader.non_binary:
                total_savings = self.total_savings(ys, y_preds)
        return sum_loss/total, correct/total, prec/j, total_savings, conf_mat

    def infer(self, binary=False):
        if self.model_type == 'torch':
            i = 0
            # infered = np.array(len(self.loader.ds))
            infered = []
            with torch.no_grad():
                for x in tqdm(self.loader.ds):
                    # print(x)
                    x = x.to(self.device)

                    pred = self.model(x)
                    # print(i, pred)
                    # infered.append(torch.argmax(pred, dim=1).cpu().numpy())
                    infered.append(pred.cpu().numpy())
                    i += 1
                    if i >= len(self.loader):
                    # if i >= 500:
                        break
        else:
            self.load_scikit_model()
            try:
                infered = self.model.predict_proba(self.loader.ds.dataset.X)
                # infered = np.argmax(infered, axis=1)
            except:
                infered = self.model.predict(self.loader.ds.dataset.X)

            # print(infered[:50])
            # try:
            #     infered = self.model.predict_proba(self.loader.dataset.ds.X)
            # except:
            #     infered = self.model.predict(self.X)
        # print(infered)
        # infered = pd.DataFrame(np.array(infered).reshape(16, 3000))
        if self.loader.binary:
            infered = np.array(infered).reshape(120000, 2)
        elif self.loader.non_binary:
            infered = np.array(infered).reshape(120000, 6)
        else:
            infered = np.array(infered).reshape(120000, 7)
        pd.DataFrame(infered).to_csv('infered.csv', index=False)
        return infered

    def get_scikit_model(self, param):
        # Bayesian
        if self.model_name == 'gaussian_nb':
            self.model = GaussianNB(**param)
        elif self.model_name == 'multi_nb':
            self.model = MultinomialNB(**param)
        elif self.model_name == 'bern_nb':
            self.model = BernoulliNB(**param)
        # Linear
        elif self.model_name == 'glm':
            self.model = LogisticRegression(**param)
        elif self.model_name == 'sgd':
            if self.loader.cross_valid:
                self.model = SGDClassifier(**param)
            else:
                raise AttributeError('SGD is implemented as the linear model for cross validation, Reload Dataloader with cross_valid = True')
        # Neighbors
        elif self.model_name == 'knn':                                     # Partial Fit Missing!!!
            self.model = KNeighborsClassifier(**param)
            # if not self.loader.cross_valid:
            # else:
            #     raise AttributeError('KNN cannot do a partial_fit, Reload Dataloader with cross_valid = False')
        # Support Vector Machines
        elif self.model_name == 'svm':
            self.model = SVC(**param)
        # Ensemble
        elif self.model_name == 'r_forest':
            self.model = RandomForestClassifier(**param)
        elif self.model_name == 'voting':
            self.model = VotingClassifier(**param)
        # Discriminant Analysis
        elif self.model_name == 'qda':
            self.model = QuadraticDiscriminantAnalysis(**param)
        elif self.model_name == 'lda':
            self.model = LinearDiscriminantAnalysis(**param)
        # Unsupervised Clustering
        elif self.model_name == 'kmeans':
            self.model = KMeans(**param)

    def load_scikit_model(self):
        self.model = pickle.load(open(os.path.join(self.output_dir, 'models', f'{self.model_name}.pkl'), 'rb'))

    def calc_sicikit_metrics(self, Y, Y_preds):
        # print([(Y[str(i)] != 0).sum() for i in Y.columns], Y.shape, Y_preds.shape)
        loss = self.loss_fn(Y, Y_preds, labels=sorted(np.unique(Y)))
        try:
            avg_prec = balanced_accuracy_score(Y, np.argmax(Y_preds, axis=1))
            f_1 = f1_score(Y, np.argmax(Y_preds, axis=1), average='weighted')
        except:
            avg_prec = balanced_accuracy_score(Y, Y_preds)
            f_1 = f1_score(Y, Y_preds, average='weighted')
        return loss, avg_prec, f_1

    def fit_scikit(self, X_train, X_test, Y_train, Y_test):
        # Fit Model
        self.model.fit(X_train, Y_train)
        pickle.dump(self.model, open(os.path.join(self.output_dir, 'models', f'{self.model_name}.pkl'), 'wb'))
        # joblib.dump(self.model, f'{self.model_name}.pkl')
        # Predict Values
        try:
            Y_train_preds = self.model.predict_proba(X_train)
            Y_test_preds = self.model.predict_proba(X_test)
        except:
            Y_train_preds = self.model.predict(X_train)
            Y_test_preds = self.model.predict(X_test)
        # print(Y_train_preds, Y_test_preds)
        # Calculate Loss / Avg Precision
        train_loss, train_avg_prec, train_f1 = self.calc_sicikit_metrics(Y_train, Y_train_preds)
        test_loss, test_avg_prec, test_f1 = self.calc_sicikit_metrics(Y_test, Y_test_preds)
        return train_loss, test_loss, train_avg_prec, test_avg_prec, train_f1, test_f1

    def train_scikit_model(self, param, param_name=''):
        # Load a New Model with the given parameters
        self.get_scikit_model(param)
        if not self.loader.cross_valid:
            # Use DataFrame instead of pytorch Loader [Probabilities Version]
            X_train, X_valid, Y_train, Y_valid = train_test_split(self.loader.train_ds.dataset.X,
                                                                  self.loader.train_ds.dataset.Y_single,
                                                                  test_size=self.loader.test_size,
                                                                  random_state=self.seed)
            # Fit Model And Calculate Losses for train and validation
            train_loss, valid_loss, train_avg_prec, valid_avg_prec, train_f1, valid_f1 = self.fit_scikit(X_train, X_valid, Y_train, Y_valid)
            # # Select Test Set [by pre-specified indices]
            X_test, Y_test = self.loader.test_ds.dataset.X.iloc[self.loader.test_ds.indices, :], self.loader.test_ds.dataset.Y_single[self.loader.test_ds.indices]
            # Predict Test Set [Unseen Data]
            try:
                Y_test_preds = self.model.predict_proba(X_test)
            except:
                Y_test_preds = self.model.predict(X_test)
            # Calculate Loss / Average Precision for Test Set
            test_loss, test_avg_prec, test_f1 = self.calc_sicikit_metrics(Y_test, Y_test_preds)
        else:
            # Use DataFrame instead of pytorch Loader [Single Label Version]
            X, Y = self.loader.train_ds.dataset.X, self.loader.train_ds.dataset.Y_single
            # Fit Model And Calculate Losses for train and validation
            results = cross_validate(self.model, X, Y, cv=self.loader.kfolds,
                                     scoring=('neg_log_loss', 'balanced_accuracy', 'f1_weighted'),
                                     return_train_score=True, n_jobs=-1)
            train_loss, valid_loss, = results['train_neg_log_loss'], results['test_neg_log_loss']
            train_avg_prec, valid_avg_prec = results['train_balanced_accuracy'], results['test_balanced_accuracy']
            train_f1, valid_f1 = results['train_f1_weighted'], results['test_f1_weighted']

            # Select Test Set [by pre-specified indices]
            X_test, Y_test = self.loader.test_ds.dataset.X.iloc[self.loader.test_ds.indices,:],\
                self.loader.test_ds.dataset.Y_single[self.loader.test_ds.indices]
            # Predict Test Set [Unseen Data]
            Y_test_preds = cross_val_predict(self.model, X_test, Y_test, cv=self.loader.kfolds, method='predict_proba')
            # Calculate Loss / Average Precision for Test Set
            test_loss, test_avg_prec, test_f1 = self.calc_sicikit_metrics(Y_test, Y_test_preds)
        # Calculate Confusion Matrix
        conf_matrix = self.confusion_matrix(Y_test, Y_test_preds)
        # Save Confusion Matrix
        if param_name:
            if isinstance(param_name, dict):
                param_name = '_'.join([f'{k}_{v}' for k, v in param_name.items()])
            self.save_df(conf_matrix, prefix=self.loader.features, suffix=f'confusion_{param_name}')
        else:
            self.save_df(conf_matrix, prefix=self.loader.features, suffix=f'confusion')

        if not self.loader.cross_valid:
            return train_loss, valid_loss, 0, train_avg_prec, valid_avg_prec, 0, train_f1, train_f1, 0
        else:
        # Average The Losses over the folds
            return np.abs(train_loss).sum()/self.loader.kfolds, np.abs(valid_loss).sum()/self.loader.kfolds, 0,\
                train_avg_prec.sum()/self.loader.kfolds, valid_avg_prec.sum()/self.loader.kfolds, 0,\
                train_f1.sum()/self.loader.kfolds, train_f1.sum()/self.loader.kfolds, 0

    def explore_scikit(self, hyperparameters=None, param_name=''):
        train_losses, valid_losses, test_losses = [], [], []
        train_avg_precs, valid_avg_precs, test_avg_precs = [], [], []
        train_f1s, valid_f1s, test_f1s = [], [], []
        if hyperparameters:
            with open('models_params_.json', 'w+') as f:
                params = {self.model_name: hyperparameters}
                json.dump(params, f)
        else:
            with open('models_params.json', 'r') as f:
                hyperparameters = json.load(f)
                hyperparameters = hyperparameters[self.model_name]
        for i in tqdm(range(len(hyperparameters))):
            param = hyperparameters[i]
            p_name = param_name + '_' + str(param[param_name]) if param_name else hyperparameters[i]
            train_loss, valid_loss, test_loss, train_avg_prec, valid_avg_prec, test_avg_prec, train_f1, valid_f1, test_f1 =\
                self.train_scikit_model(param, param_name=p_name)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            test_losses.append(test_loss)
            train_avg_precs.append(train_avg_prec)
            valid_avg_precs.append(valid_avg_prec)
            test_avg_precs.append(test_avg_prec)
            train_f1s.append(train_f1)
            valid_f1s.append(valid_f1)
            test_f1s.append(test_f1)
            print(train_loss, valid_loss, test_loss, train_avg_prec, valid_avg_prec, test_avg_prec, train_f1, valid_f1, test_f1)

        # Create Results DF
        results = pd.DataFrame({'Train/Loss': train_losses, 'Valid/Loss': valid_losses, 'Test/Loss': test_losses,
                                'Train/Precision': train_avg_precs, 'Valid/Precision': valid_avg_precs, 'Test/Precision': test_avg_precs,
                                'Train/F1': train_f1s, 'Valid/F1': valid_f1s, 'Test/F1': test_f1s})
        self.save_df(results, prefix=self.loader.features)
        return results

    def board_update(self, i, train_loss, train_prec, train_save, val_loss, val_prec, valid_save, train_conf_mat, valid_conf_mat, k=0):
        if not self.loader.cross_valid:
            self.board_writer.add_scalar('Loss/Train', train_loss, i)
            self.board_writer.add_scalar('Loss/Valid', val_loss, i)
            self.board_writer.add_scalar('Precision/Train', train_prec, i)
            self.board_writer.add_scalar('Precision/Valid', val_prec, i)
            self.board_writer.add_scalar('Save/Train', train_save, i)
            self.board_writer.add_scalar('Save/Valid', valid_save, i)

            self.board_writer.add_figure("Conf/Train", self.plot_conf(train_conf_mat), i)
            self.board_writer.add_figure("Conf/Valid", self.plot_conf(valid_conf_mat), i)
        else:
            self.board_writers[k].add_scalar('Loss/Train', train_loss, i)
            self.board_writers[k].add_scalar('Loss/Valid', val_loss, i)
            self.board_writers[k].add_scalar('Precision/Train', train_prec, i)
            self.board_writers[k].add_scalar('Precision/Valid', val_prec, i)
            self.board_writers[k].add_scalar('Save/Train', train_save, i)
            self.board_writers[k].add_scalar('Save/Valid', valid_save, i)

            self.board_writers[k].add_figure("Conf/Train", self.plot_conf(train_conf_mat), i)
            self.board_writers[k].add_figure("Conf/Valid", self.plot_conf(valid_conf_mat), i)
        # for name, weight in self.model.named_parameters():
        #     self.board_writer.add_histogram(f'{name}.grad', weight.grad, i)

    def board_update_hp(self, epochs, train_loss, train_prec, train_save, val_loss, val_prec, val_save, test_loss, test_prec, test_save, k=0):
        if not self.loader.cross_valid:
            lr = self.optimizer.param_groups[0]['lr']
            self.board_writer.add_hparams(
                {"lr": lr, "batch_size": self.loader.train.batch_size, "epochs": epochs},
                {"Loss/Train": train_loss, "Loss/Valid": val_loss, "Loss/Test": test_loss,
                 "Precision/Train": train_prec, "Precision/Valid": val_prec, "Precision/Test": test_prec,
                 "Save/Train": train_save, "Save/Valid": val_save, "Save/Test": test_save,}
            ,)
        else:
            lr = self.optimizers[k].param_groups[0]['lr']
            self.board_writers[k].add_hparams(
                {"lr": lr, "batch_size": self.loader.train_folds[k].batch_size, "epochs": epochs},
                {"Loss/Train": train_loss, "Loss/Valid": val_loss, "Loss/Test": test_loss,
                 "Precision/Train": train_prec, "Precision/Valid": val_prec, "Precision/Test": test_prec,
                 "Save/Train": train_save, "Save/Valid": val_save, "Save/Test": test_save,}
            )

    def resume(self, resume_training):
        if resume_training:
            self.log_name = resume_training[:-4]
            self.model_path = os.path.join(self.model_dir, resume_training)
            if self.model_type == 'torch':
                if not self.loader.cross_valid:
                    checkpoint = torch.load(self.model_path + '.pt')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if self.optimizer:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.optimizer_to(self.optimizer)
                    self.model.eval()
                else:
                    for k in range(self.loader.kfolds):
                        checkpoint = torch.load(self.model_paths[k] + '.pt')
                        self.models[k].load_state_dict(checkpoint['model_state_dict'])
                        if self.optimizers:
                            self.optimizers[k].load_state_dict(checkpoint['optimizer_state_dict'])
                            self.optimizer_to(self.optimizers[k])
                        self.models[k].eval()
            else:
                # Just for the sake of completion
                load(self.model_path + '.joblib')

    def optimizer_to(self, optimizer):
        for param in optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

    def save_checkpoint(self, epoch, model, optimizer, k=0):
        if self.model_type == 'torch':
            if not self.loader.cross_valid:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, self.model_path[:-3] + f'_{str(epoch).zfill(3)}.pt')
                self.model_path = self.model_path[:-3] + f'{str(epoch).zfill(3)}'
            else:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, self.model_paths[k][:-3] + f'_{str(epoch).zfill(3)}.pt')
                self.model_paths[k] = self.model_paths[k][:-3] + f'{str(epoch).zfill(3)}'
        else:
            # Just for the sake of completion
            dump(self.model, self.model_path[:-3] + f'_{str(epoch).zfill(3)}.joblib')
            self.model_path = self.model_path[:-3] + f'{str(epoch).zfill(3)}'

    def save_df(self, df, prefix='', suffix=''):
        if isinstance(prefix, list):
            prefix = '_'.join(prefix)
            if len(prefix) > 100:
                prefix = prefix[:100]
        if isinstance(suffix, dict):
            suffix = '_'.join([f'{k}_{v}' for k, v in enumerate(suffix)])
        if prefix and suffix:
            df.to_csv(f'{os.path.join(self.report_model_dir, prefix.replace("*", "all"))}_{self.model_name}_{suffix}.csv', index=False)
        elif prefix and not suffix:
            df.to_csv(f'{os.path.join(self.report_model_dir, prefix.replace("*", "all"))}_{self.model_name}.csv', index=False)
        elif not prefix and suffix:
            df.to_csv(f'{os.path.join(self.report_model_dir, self.model_name)}_{suffix}.csv', index=False)
        else:
            df.to_csv(f'{os.path.join(self.report_model_dir, self.model_name)}.csv', index=False)

    def plot_conf(self, conf, classes=7):
        # df_cm = pd.DataFrame(conf / np.sum(conf, axis=1)[:, None],
        #                      index=[i for i in classes],
        #                      columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        return sns.heatmap(conf.cpu().numpy(), annot=True).get_figure()