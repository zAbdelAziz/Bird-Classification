import json
import os, sys, warnings

import numpy as np
import pandas as pd

from dataset import Loader, AudioDS
from classifier import Classifier
from models import *

from tqdm import tqdm

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


def get_features(model_name):
    with open('models_features.json', 'r') as f:
        features = json.load(f)
    r_features = features[model_name]['reduce']
    features = features[model_name]['features']
    return features, r_features


def majority_vote(call):
    # Get the counts of each unique element
    counts = np.bincount(call)
    # Get highest and second highest frequencies
    votes = counts.argsort()
    max_vote, sec_vote = votes[-1], votes[-2]
    # Replace 2 with the second highest frequency if exists [Or should it be zeros]?
    if max_vote == 2 and counts.sum() != counts[2]:
        max_vote = sec_vote
    return max_vote


def smooth_output(df):
    df = df.iloc[:,1:]
    print(df)
    for r in range(df.shape[0]):
        # Select each row
        row = df.iloc[r, :].values
        nonzero_indices = np.where(row != 0)[0]
        subarrays = []
        subarray_start = nonzero_indices[0]
        # Iterate over nonzero indices
        for i in range(1, len(nonzero_indices)):
            subarray_end = nonzero_indices[i - 1] + 1
            if nonzero_indices[i] != subarray_end:
                subarray = row[subarray_start:subarray_end].copy()
                if len(subarray) > 2:
                    vote = majority_vote(subarray)
                    df.iloc[r, subarray_start:subarray_end] = vote
                subarrays.append(subarray)
                subarray_start = nonzero_indices[i]

        # Add the last subarray
        subarray = row[subarray_start:nonzero_indices[-1] + 1]
        subarrays.append(subarray)

    df.to_csv('infered_smoothed.csv', index=False)


def main(mode='explore', model_type='torch'):
    if mode == 'explore':
        feature_sets = ['raw_melspect_mean', 'raw_melspect_std', 'cln_melspect_mean', 'cln_melspect_std',
                         'raw_mfcc_mean', 'raw_mfcc_std', 'cln_mfcc_mean', 'cln_mfcc_std',
                         'raw_mfcc_d_mean', 'raw_mfcc_d_std', 'cln_mfcc_d_mean', 'cln_mfcc_d_std',
                         'raw_mfcc_d2_mean', 'raw_mfcc_d2_std', 'cln_mfcc_d2_mean', 'cln_mfcc_d2_std',
                         'zcr', 'yin', 'bandwidth_mean', 'bandwidth_std',
                         'flatness_mean', 'flatness_std', 'centroid_mean', 'centroid_std',
                         'flux_mean', 'flux_std', 'energy_mean', 'energy_std', 'power_mean', 'power_std',
                         'raw_contrast_mean', 'raw_contrast_std', 'cln_contrast_mean', 'cln_contrast_std',
                         ]

        model_type = 'scikit'
        model_name = 'r_forest'

        features, reduce_features = get_features(model_name)
        # loader = Loader(features=features, binary=True)
        # loader = Loader(features=features, reduce_features=reduce_features, reducer=model_name, binary=True)
        loader = Loader(features=features, reduce_features=reduce_features, reducer=model_name)
        # loader = Loader(binary=True)

        classifier = Classifier(loader, model_type=model_type, model_name=model_name)
        classifier.explore()

    elif mode == 'train':
        # model_name = 'cnn_1d_basic'
        model_name = 'linear_basic'

        features, reduce_features = get_features(model_name)
        # loader = Loader(features=features, reduce_features=reduce_features, reducer=model_name,
        #                 cross_valid=True, kfolds=2,
        #                 batch_size=64, pin_memory=True, num_workers=4, shuffle=True)

        # loader = Loader(features=features, reducer=model_name,
        #                 cross_valid=True, kfolds=3,
        #                 batch_size=64, pin_memory=True, num_workers=4)

        loader = Loader(features=features, reducer=model_name,
                        # cross_valid=True, kfolds=3,
                        x_type='single',
                        shuffle=True,
                        batch_size=64, pin_memory=True, num_workers=4)

        # LINEAR MODEL [Simple]
        model = SimpleFFNN(_in=loader.train_ds.dataset.X.shape[-1], _out=7, activation=ReLU(), softmax=True)
        # model = SimpleFFNN(_in=loader.train_ds.dataset.X.shape[-1], _out=2, activation=ReLU())

        # CONV 1D [Simple]
        # model = SimpleCNN(_in=1, _out=7, n_layers=8, n_filters=128, d_1=True)
        # model = SimpleCNN(_in=1, _out=2, n_layers=8, n_filters=128, d_1=True)
        # model = SimpleCNN(_in=1, _out=6, n_layers=8, n_filters=128, d_1=True)

        # model = SimpleCNN(_in=1, _out=7, n_layers=4, n_filters=512, d_1=True, enc=True, activation=True)

        # CONV 1D [AutoEncoder]
        # model = FunnelCNNAutoEnc(_out=7, leaky=True, activation=ReLU())

        # model = VGG()
        # model = ResNet(ResidualBlock, [3, 4, 6, 3])
        # model = InceptionNet(7)

        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # loss_fn = CrossEntropyLoss()
        label_weights = torch.tensor(loader.ds.class_weights.copy()).to(torch.device('cuda:0'))
        loss_fn = CrossEntropyLoss(weight=label_weights)

        classifier = Classifier(loader, model=model, loss_fn=loss_fn, optimizer=optimizer,
                                model_name=model_name, model_type=model_type)
        classifier.train(20)
        # classifier.infer()
    elif mode == 'infer':
        model_type = 'torch'
        # model_name = 'linear_basic'
        model_name = 'cnn_1d_basic'

        # model_type = 'scikit'
        # model_name = 'knn'

        features, reduce_features = get_features(model_name)
        # loader = Loader(mode='infer', features=features, reduce_features=reduce_features, reducer=model_name,
        #                 balance_dataset=False, batch_size=1)
        loader = Loader(mode='infer', features=features, binary=True,
                        balance_dataset=False, batch_size=1)

        # LINEAR MODEL [Simple]
        # model = SimpleFFNN(_in=loader.ds.dataset.X.shape[-1], _out=7, activation=ReLU())
        # model = SimpleFFNN(_in=loader.ds.dataset.X.shape[-1], _out=2, activation=ReLU(), softmax=True)

        # CONV 1D [Simple]
            # Step 3 [Best but overfitting]
        # model = SimpleCNN(_in=1, _out=2, n_layers=8, n_filters=128, d_1=True)
        # model = SimpleCNN(_in=1, _out=7, n_layers=8, n_filters=128, d_1=True)
            # Step 16 [Best]
        # model = SimpleCNN(_in=1, _out=7, n_layers=4, n_filters=512, d_1=True, enc=True)

        # CONV 1D [AutoEncoder]
            # Step 2 [Best So Far]
        model = FunnelCNNAutoEnc(leaky=True)


        # Classifier
        classifier = Classifier(loader, model=model, model_name=model_name, model_type=model_type,
                                mode='infer', resume='05-06-2023 22_51_017')
        # classifier = Classifier(loader, model_name=model_name, model_type=model_type, mode='infer')

        classifier.infer()


if __name__ == "__main__":
    # main(mode='explore')
    main(mode='train')
    # main(mode='infer')

