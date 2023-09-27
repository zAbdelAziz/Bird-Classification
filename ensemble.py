import os, sys, warnings

import numpy as np
import pandas as pd

from dataset import *
from classifier import *
from models import *

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


class Ensemble:
    # def __init__(self, classifiers=('nr_cnn1', 'nr_cae', 'nr_cnn2')):
    def __init__(self, multi_stage=False, b_classifiers=None, nb_classifiers=None, classifiers=None):

        self.multi_stage = multi_stage

        self.b_classifiers, self.nb_classifiers, self.classifiers = {}, {}, {}

        self.get_loaders()

        if not b_classifiers is None:
            for classifier in b_classifiers:
                print(classifier)
                self.b_classifiers[classifier] = self.get_classifier(classifier, 2)
            print(self.b_classifiers)

        if not nb_classifiers is None:
            for classifier in nb_classifiers:
                print(classifier)
                self.nb_classifiers[classifier] = self.get_classifier(classifier, 6)
            print(self.nb_classifiers)

        if not classifiers is None:
            for classifier in classifiers:
                print(classifier)
                self.classifiers[classifier] = self.get_classifier(classifier, 7)
            print(self.classifiers)

    def get_loaders(self):
        self.lin_features, self.lin_features_r = self.get_features('linear_basic')
        self.lin_loader = Loader(mode='infer', features=self.lin_features, reducer='linear_basic', x_type='single', batch_size=64)
        # self.lin_loader_binary = Loader(mode='infer', features=self.lin_features, binary=True, reducer='linear_basic', x_type='single', batch_size=64)
        # self.lin_loader_nonbinary = Loader(mode='infer', features=self.lin_features, non_binary=True, reducer='linear_basic', x_type='single', batch_size=64)
        # self.cnn_features, self.cnn_features_r = self.get_features('cnn_1d_basic')
        # self.cnn_loader = Loader(mode='infer', features=self.cnn_features, reducer='cnn_1d_basic', batch_size=64)
        # self.cnn_loader_binary = Loader(mode='infer', features=self.cnn_features, binary=True, reducer='cnn_1d_basic', batch_size=64)
        # self.cnn_loader_nonbinary = Loader(mode='infer', features=self.cnn_features, non_binary=True, reducer='cnn_1d_basic', batch_size=64)

    def get_classifier(self, model_name, _out):
        if 'cnn1' in model_name:
            model = SimpleCNN(_in=1, _out=_out, n_layers=8, n_filters=128, d_1=True)
        elif 'cnn2' in model_name:
            model = SimpleCNN(_in=1, _out=_out, n_layers=4, n_filters=512, d_1=True, enc=True, activation=True)
        elif 'cnn3' in model_name:
            model = FunnelCNNAutoEnc(_out=_out, leaky=True, activation=ReLU())
        elif 'sffnn' in model_name:
            model = SimpleFFNN(_in=241, _out=7, activation=ReLU(), batch_normal=True, softmax=True)
        elif model_name == 'vgg':
            model = VGG()
        elif model_name == 'resnet':
            model = ResNet(ResidualBlock, [3, 4, 6, 3])
        elif model_name == 'incept':
            model = InceptionNet(_out)
        else:
            raise NotImplementedError('Model not supported')

        if _out == 2:
            if 'cnn' in model_name:
                return Classifier(self.cnn_loader_binary, model=model, model_name='cnn_1d_basic', model_type='torch',
                                  mode='infer', resume=model_name)
            else:
                return Classifier(self.lin_loader_binary, model=model, model_name='linear_basic', model_type='torch',
                                  mode='infer', resume=model_name)
        elif _out == 6:
            if 'cnn' in model_name:
                return Classifier(self.cnn_loader_nonbinary, model=model, model_name='cnn_1d_basic', model_type='torch',
                                  mode='infer', resume=model_name)
            else:
                return Classifier(self.lin_loader_nonbinary, model=model, model_name='linear_basic', model_type='torch',
                                  mode='infer', resume=model_name)
        else:
            if 'cnn' in model_name:
                return Classifier(self.cnn_loader, model=model, model_name='cnn_1d_basic', model_type='torch',
                                  mode='infer', resume=model_name)
            else:
                return Classifier(self.lin_loader, model=model, model_name='linear_basic', model_type='torch',
                                  mode='infer', resume=model_name)

    @staticmethod
    def get_features(model_name):
        with open('models_features.json', 'r') as f:
            features = json.load(f)
        r_features = features[model_name]['reduce']
        features = features[model_name]['features']
        return features, r_features

    @staticmethod
    def majority_vote(arr, best=1):
        # Get the counts of each unique element
        counts = np.bincount(arr)
        # Get highest frequency
        votes = counts.argsort()[-1]
        return votes

    def infer(self):
        if len(self.classifiers) > 0:
            infered_all = self.infer_classifier(self.classifiers)
            infered_all.to_csv('infered_all.csv', index=False)
        # if len(self.b_classifiers) > 0:
        #     infered_b = self.infer_classifier(self.b_classifiers)
        #     infered_b.to_csv('infered_bin.csv', index=False)


        # if self.multi_stage:
        #     # Stage 1
        #     b_votes = pd.DataFrame(columns=self.b_classifiers.keys())
        #     for k, b_classifier in self.b_classifiers.items():
        #         b_infered = b_classifier.infer()
        #         b_votes[k] = np.argmax(b_infered, axis=1)
        #     b_votes.to_csv('b_infered_votes.csv', index=False)
        #     print(b_votes)
        #     b_voted = []
        #     for i in range(len(b_votes)):
        #         b_voted.append(self.majority_vote(b_votes.iloc[i, :].values))
        #     b_voted = pd.DataFrame(np.array(b_voted))
        #     print(b_voted)
        #     b_voted.to_csv('b_infered_voted.csv', index=False)
        #     # Stage 2
        #     nb_votes = pd.DataFrame(columns=self.classifiers.keys())
        #     for k, nb_classifier in self.classifiers.items():
        #         nb_infered = nb_classifier.infer()
        #         nb_votes[k] = np.argmax(nb_infered, axis=1)
        #     nb_votes.to_csv('nb_infered_votes.csv', index=False)
        #     print(nb_votes)
        #     nb_voted = []
        #     for i in range(len(nb_votes)):
        #         nb_voted.append(self.majority_vote(nb_votes.iloc[i, :].values))
        #     nb_voted = pd.DataFrame(np.array(nb_voted))
        #     print(nb_voted)
        #     nb_voted.to_csv('nb_infered_voted.csv', index=False)
        #     # Stage 3
        #     # b_voted = pd.read_csv('b_infered_voted.csv')
        #     # nb_voted = pd.read_csv('nb_infered_voted.csv')
        #     votes = nb_voted.copy()
        #     nb_nonzeros = np.where(nb_voted != 0)[0]
        #     zeros = np.where(b_voted == 0)[0]
        #     to_ = set(nb_nonzeros) & set(zeros)
        #     print(to_, type(to_))
        #     votes.iloc[list(to_),:] = 0
        #     # print(nb_voted.loc[nb_voted.values != 0])
        #     # print((b_voted.loc[nb_nonzeros, :].values == 0).index)
        #     # print(votes.loc[b_voted.loc[nb_voted != 0, :], :])
        #     # votes.loc[b_voted.loc[nb_voted.loc[nb_voted.values != 0], :], :] = 0
        #     # zeros = np.where(b_voted.iloc[nb_nonzeros,:] == 0)[0]
        #     # print('', len(zeros), zeros.tolist())
        #     # votes.iloc[zeros, :] = 0
        #     # votes = pd.DataFrame(votes)
        #     print(votes)
        #     votes.to_csv('infered.csv', index=False)
        # else:
        #     votes = pd.DataFrame(columns=self.classifiers.keys())
        #     voted = []
        #     for k, classifier in self.classifiers.items():
        #         infered = classifier.infer()
        #         votes[k] = np.argmax(infered, axis=1)
        #     print(votes)
        #     votes.to_csv('infered_votes.csv', index=False)
        #     # votes = pd.read_csv('infered_votes.csv')
        #     for i in range(len(votes)):
        #         voted.append(self.majority_vote(votes.iloc[i, :].values))
        #     voted = pd.DataFrame(np.array(voted))
        #     print(voted)
        #     voted.to_csv('infered_voted.csv', index=False)
        #     # votes = pd.read_csv('infered_votes.csv')
        #     # nb_votes = pd.read_csv('infered_voted.csv')

    def infer_classifier(self, classifiers):
        votes = pd.DataFrame(columns=classifiers.keys())
        for k, classifier in classifiers.items():
            infered = classifier.infer()
            votes[k] = np.argmax(infered, axis=1)
        votes.to_csv('infered_votes.csv', index=False)
        print(votes)
        voted = []
        for i in range(len(votes)):
            voted.append(self.majority_vote(votes.iloc[i, :].values))
        voted = pd.DataFrame(np.array(voted))
        print(voted)
        return voted


# classifiers = ((f'cnn1_0{i+1}' for i in range(8)))
classifiers = ((['sffnn_1']))
# b_classifiers = (['cnn1_b_1'])
e = Ensemble(classifiers=classifiers)
e.infer()