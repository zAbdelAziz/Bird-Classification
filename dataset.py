import os, json
import pickle
from glob import glob

from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

import torch
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset, random_split, sampler

from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AudioDS(Dataset):
    def __init__(self, data_dir='data', output_dir='outputs', analyze=False, from_files=False, mode='train',
                 win_size=2, plot=True, binary=False, non_binary=False, gain_weight=False, percent_weights=False,
                 features=('*'), scale_X=True, balance_dataset=False, reduce_features=[], reducer='model', x_type='1d'):
        self.mode, self.binary, self.non_binary = mode, binary, non_binary
        self.labels_ids, self.labels_names, self.label_files, self.data_files, self.df_files = \
            None, None, None, None, None

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.from_files = from_files

        if self.mode == 'train':
            # Load Dataset [Single DF / From Files]
            self.X, self.Y_raw, self.Y_normalized = None, None, None
            self.load_data(self.from_files)
            if scale_X:
                xmin = pd.DataFrame(self.X.min().values).transpose()
                xmax = pd.DataFrame(self.X.max().values).transpose()
                xmin.columns, xmax.columns = self.X.columns, self.X.columns
                if self.binary:
                    xmin.to_csv('data/xmin_binary.csv')
                    xmax.to_csv('data/xmax_binary.csv')
                elif self.non_binary:
                    xmin.to_csv('data/xmin_nonbinary.csv')
                    xmax.to_csv('data/xmax_nonbinary.csv')
                else:
                    xmin.to_csv('data/xmin.csv')
                    xmax.to_csv('data/xmax.csv')
                self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())

            self.Y_OneHot = self.create_onehot_enc()
            self.Y_single = np.argmax(self.Y_normalized.values, axis=1)

            # Binary
            if self.binary:
                # TODO Select Threshold [VV Important]
                # self.Y_normalized.loc[self.Y_normalized['0'] >= 0.5, '0'] = 1
                # self.Y_normalized.loc[self.Y_normalized['0'] < 0.5, '1'] = 1
                self.Y_normalized.iloc[:, 1] = self.Y_normalized.iloc[:, 1:].sum(axis=1)
                self.Y_normalized.drop(['2', '3', '4', '5', '6'], axis=1, inplace=True)
                # self.Y_normalized = self.Y_normalized.round()
                self.Y_single[np.nonzero(self.Y_single)] = 1
                # self.Y_normalized = self.Y_normalized.round()
                print(self.Y_normalized.head(50))
                # print(self.Y_normalized.iloc[-100:-50,:])
            # Nonbinary
            elif self.non_binary:
                ind = self.Y_normalized.loc[self.Y_normalized['0'] < 0.5].index
                self.Y_normalized = self.Y_normalized.iloc[ind, 1:]
                self.Y_OneHot = self.Y_OneHot.iloc[ind, 1:]
                self.Y_single = self.Y_single[ind]
                self.X = self.X.iloc[ind, :]

                self.Y_normalized.reset_index(inplace=True, drop=True)
                self.Y_OneHot.reset_index(inplace=True, drop=True)
                self.X.reset_index(inplace=True, drop=True)
            # All
            # else:
            #     # self.Y_normalized = self.Y_normalized.round()
            #     # self.Y_OneHot = self.Y_OneHot.round()
            #     # self.Y_OneHot = self.Y_OneHot

            # Balance Dataset Classes
            self.create_class_weights(gain_weight=gain_weight, percent=percent_weights)
            # if balance_dataset:
            #     # self.create_class_weights(gain_weight=gain_weight)
            #     self.balance_dataset()

        elif self.mode == 'infer':
            self.X, self.Y_raw, self.Y_normalized = None, None, None
            self.load_data_infer()
            # Scale Data
            if scale_X:
                if self.binary:
                    xmin_name = f'data/xmin_binary.csv'
                    xmax_name = f'data/xmax_binary.csv'
                elif self.non_binary:
                    xmin_name = f'data/xmin_nonbinary.csv'
                    xmax_name = f'data/xmax_nonbinary.csv'
                else:
                    xmin_name = f'data/xmin.csv'
                    xmax_name = f'data/xmax.csv'
                xmin = pd.read_csv(xmin_name).transpose().squeeze()
                xmax = pd.read_csv(xmax_name).transpose().squeeze()
                self.X = (self.X - xmin) / (xmax - xmin)

        # Calculate Window Size
        self.win_size = win_size

        # Get Feature Names
        if not from_files:
            self.feature_sets, self.features = [], {}
            self.load_feature_sets()

        self.labels_analysis, self.features_analysis = None, None
        if analyze:
            self.plot = plot
            if from_files:
                self.load_data(from_files=False)
            self.analyze_dataset()

        # Select Feature Sets
        if '*' in features:
            self.X = self.select_features(self.feature_sets)
        else:
            self.X = self.select_features(features)

        # Reduce Feature Sets
        for f in reduce_features:
            self.reducer = reducer
            # Select All Features of the feature set
            to_reduce = [self.features[f]]
            x = pd.concat([self.X[ft] for ft in self.features[f]], 1)
            # Reduce Using PCA
            if self.binary:
                r_name = f'binary_{self.reducer}_{f}.pkl'
            elif self.non_binary:
                r_name = f'nonbinary_{self.reducer}_{f}.pkl'
            else:
                r_name = f'{self.reducer}_{f}.pkl'

            try:
                pca = pickle.load(open(os.path.join(self.output_dir, 'reducers', r_name), 'rb'))
                reduced = pca.transform(x)
                print(f'reduced {f} by loader')
            except:
                pca = PCA(n_components=len(to_reduce[0])//4)
                reduced = pca.fit_transform(x)
                pickle.dump(pca, open(os.path.join(self.output_dir, 'reducers', r_name), 'wb'))
                print(f'created new reducer for {f}')
            # Rename columns and Append to oroginal data
            col_names = [f'{f}_{str(i)}' for i in range(reduced.shape[1])]
            reduced = pd.DataFrame(reduced, columns=col_names)
            cols = [c for c in self.X.columns.values.tolist() if c not in x.columns.values.tolist()]
            cols.extend(col_names)
            self.X.drop(x.columns.values.tolist(), axis=1, inplace=True)
            self.X = pd.concat([self.X, reduced], axis=1, ignore_index=True)
            self.X.columns = cols

        self.x_type = x_type

        # # Class Weights
        # if self.mode == 'train':
        #     self.class_weights = []
        #     classes = self.Y_normalized.columns
        #     # for c in classes:
        #     #     self.class_weights.append((self.Y_normalized[c] == 1).sum() / len(self.Y_normalized))
        #     # self.class_weights = np.array(self.class_weights)
        #     # print(self.class_weights)
        #     if self.binary:
        #         for c in classes:
        #             self.class_weights.append((self.Y_normalized[c] >= 0.5).sum() / len(self.Y_normalized))
        #         self.class_weights = np.array(self.class_weights)[::-1]
        #         # self.class_weights = self.class_weights
        #     elif self.non_binary:
        #         for c in classes:
        #             self.class_weights.append((self.Y_normalized[c] == 1).sum() / len(self.Y_normalized))
        #         self.class_weights = np.array(self.class_weights)
        #         self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
        #         self.class_weights /= self.class_weights.sum()
        #         # self.class_weights[0] *= 1.2
        #         # self.class_weights[1] *= 0.6
        #         # self.class_weights[2] *= 0.7
        #         # self.class_weights[3] *= 7
        #         # self.class_weights[4] *= 1.4
        #         # self.class_weights[5] *= 1.8
        #     else:
        #         for c in classes:
        #             self.class_weights.append((self.Y_normalized[c] == 1).sum() / len(self.Y_normalized))
        #         self.class_weights = np.array(self.class_weights)
        #         if not gain_weight:
        #             self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
        #             self.class_weights /= self.class_weights.sum()
        #             if not balance_dataset:
        #                 self.class_weights[:-3] *= 2
        #                 self.class_weights[0] *= 10
        #             # print(self.class_weights.sum())
        #             # self.class_weights = torch.nn.Softmax()(torch.tensor(self.class_weights))
        #         else:
        #             self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
        #             # self.class_weights /= self.class_weights.sum()
        #             gain_matrix = np.array([[0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
        #                                         [-0.25, 1, -0.3, -0.1, -0.1, -0.1, -0.1],
        #                                         [-0.02, -0.1, 1, -0.1, -0.1, -0.1, -0.1],
        #                                         [-0.25, -0.1, -0.3, 1, -0.1, -0.1, -0.1],
        #                                         [-0.25, -0.1, -0.3, -0.1, 1, -0.1, -0.1],
        #                                         [-0.25, -0.1, -0.3, -0.1, -0.1, 1, -0.1],
        #                                         [-0.25, -0.1, -0.3, -0.1, -0.1, -0.1, 1]])
        #             self.class_weights = np.matmul(self.class_weights, gain_matrix)
        #             self.class_weights /= self.class_weights.sum()
        #
        #             # self.class_weights = torch.nn.Softmax()(torch.tensor(self.class_weights))
        #         # self.class_weights[0] = 1
        #         # self.class_weights[1] = 3
        #         # self.class_weights[2] = 3
        #         # self.class_weights[3] = 10
        #         # self.class_weights[4] = 4
        #         # self.class_weights[5] = 4
        #
        #     print(self.class_weights.sum(), self.class_weights)

        # Reorder the columns
        # cnn_1d_basic
        if len(reduce_features) == 0:
            if 'cnn' in reducer:
                self.X = self.X.reindex(columns=['raw_melspect_mean_0', 'raw_melspect_mean_1', 'raw_melspect_mean_2', 'raw_melspect_mean_3', 'raw_melspect_mean_4', 'raw_melspect_mean_5', 'raw_melspect_mean_6', 'raw_melspect_mean_7', 'raw_melspect_mean_8', 'raw_melspect_mean_9', 'raw_melspect_mean_10', 'raw_melspect_mean_11', 'raw_melspect_mean_12', 'raw_melspect_mean_13', 'raw_melspect_mean_14', 'raw_melspect_mean_15', 'raw_melspect_mean_16', 'raw_melspect_mean_17', 'raw_melspect_mean_18', 'raw_melspect_mean_19', 'raw_melspect_mean_20', 'raw_melspect_mean_21', 'raw_melspect_mean_22', 'raw_melspect_mean_23', 'raw_melspect_mean_24', 'raw_melspect_mean_25', 'raw_melspect_mean_26', 'raw_melspect_mean_27', 'raw_melspect_mean_28', 'raw_melspect_mean_29', 'raw_melspect_mean_30', 'raw_melspect_mean_31', 'raw_melspect_mean_32', 'raw_melspect_mean_33', 'raw_melspect_mean_34', 'raw_melspect_mean_35', 'raw_melspect_mean_36', 'raw_melspect_mean_37', 'raw_melspect_mean_38', 'raw_melspect_mean_39', 'raw_melspect_mean_40', 'raw_melspect_mean_41', 'raw_melspect_mean_42', 'raw_melspect_mean_43', 'raw_melspect_mean_44', 'raw_melspect_mean_45', 'raw_melspect_mean_46', 'raw_melspect_mean_47', 'raw_melspect_mean_48', 'raw_melspect_mean_49', 'raw_melspect_mean_50', 'raw_melspect_mean_51', 'raw_melspect_mean_52', 'raw_melspect_mean_53', 'raw_melspect_mean_54', 'raw_melspect_mean_55', 'raw_melspect_mean_56', 'raw_melspect_mean_57', 'raw_melspect_mean_58', 'raw_melspect_mean_59', 'cln_melspect_mean_0', 'cln_melspect_mean_1', 'cln_melspect_mean_2', 'cln_melspect_mean_3', 'cln_melspect_mean_4', 'cln_melspect_mean_5', 'cln_melspect_mean_6', 'cln_melspect_mean_7', 'cln_melspect_mean_8', 'cln_melspect_mean_9', 'cln_melspect_mean_10', 'cln_melspect_mean_11', 'cln_melspect_mean_12', 'cln_melspect_mean_13', 'cln_melspect_mean_14', 'cln_melspect_mean_15', 'cln_melspect_mean_16', 'cln_melspect_mean_17', 'cln_melspect_mean_18', 'cln_melspect_mean_19', 'cln_melspect_mean_20', 'cln_melspect_mean_21', 'cln_melspect_mean_22', 'cln_melspect_mean_23', 'cln_melspect_mean_24', 'cln_melspect_mean_25', 'cln_melspect_mean_26', 'cln_melspect_mean_27', 'cln_melspect_mean_28', 'cln_melspect_mean_29', 'cln_melspect_mean_30', 'cln_melspect_mean_31', 'cln_melspect_mean_32', 'cln_melspect_mean_33', 'cln_melspect_mean_34', 'cln_melspect_mean_35', 'cln_melspect_mean_36', 'cln_melspect_mean_37', 'cln_melspect_mean_38', 'cln_melspect_mean_39', 'cln_melspect_mean_40', 'cln_melspect_mean_41', 'cln_melspect_mean_42', 'cln_melspect_mean_43', 'cln_melspect_mean_44', 'cln_melspect_mean_45', 'cln_melspect_mean_46', 'cln_melspect_mean_47', 'cln_melspect_mean_48', 'cln_melspect_mean_49', 'cln_melspect_mean_50', 'cln_melspect_mean_51', 'cln_melspect_mean_52', 'cln_melspect_mean_53', 'cln_melspect_mean_54', 'cln_melspect_mean_55', 'cln_melspect_mean_56', 'cln_melspect_mean_57', 'cln_melspect_mean_58', 'cln_melspect_mean_59', 'raw_mfcc_mean_0', 'raw_mfcc_mean_1', 'raw_mfcc_mean_2', 'raw_mfcc_mean_3', 'raw_mfcc_mean_4', 'raw_mfcc_mean_5', 'raw_mfcc_mean_6', 'raw_mfcc_mean_7', 'raw_mfcc_mean_8', 'raw_mfcc_mean_9', 'raw_mfcc_mean_10', 'raw_mfcc_mean_11', 'raw_mfcc_mean_12', 'raw_mfcc_mean_13', 'raw_mfcc_mean_14', 'raw_mfcc_mean_15', 'raw_mfcc_mean_16', 'raw_mfcc_mean_17', 'raw_mfcc_mean_18', 'raw_mfcc_mean_19', 'raw_mfcc_std_0', 'raw_mfcc_std_1', 'raw_mfcc_std_2', 'raw_mfcc_std_3', 'raw_mfcc_std_4', 'raw_mfcc_std_5', 'raw_mfcc_std_6', 'raw_mfcc_std_7', 'raw_mfcc_std_8', 'raw_mfcc_std_9', 'raw_mfcc_std_10', 'raw_mfcc_std_11', 'raw_mfcc_std_12', 'raw_mfcc_std_13', 'raw_mfcc_std_14', 'raw_mfcc_std_15', 'raw_mfcc_std_16', 'raw_mfcc_std_17', 'raw_mfcc_std_18', 'raw_mfcc_std_19'])
            elif 'linear' in reducer:
                self.X = self.X.reindex(columns=['raw_melspect_mean_0', 'raw_melspect_mean_1', 'raw_melspect_mean_2', 'raw_melspect_mean_3', 'raw_melspect_mean_4', 'raw_melspect_mean_5', 'raw_melspect_mean_6', 'raw_melspect_mean_7', 'raw_melspect_mean_8', 'raw_melspect_mean_9', 'raw_melspect_mean_10', 'raw_melspect_mean_11', 'raw_melspect_mean_12', 'raw_melspect_mean_13', 'raw_melspect_mean_14', 'raw_melspect_mean_15', 'raw_melspect_mean_16', 'raw_melspect_mean_17', 'raw_melspect_mean_18', 'raw_melspect_mean_19', 'raw_melspect_mean_20', 'raw_melspect_mean_21', 'raw_melspect_mean_22', 'raw_melspect_mean_23', 'raw_melspect_mean_24', 'raw_melspect_mean_25', 'raw_melspect_mean_26', 'raw_melspect_mean_27', 'raw_melspect_mean_28', 'raw_melspect_mean_29', 'raw_melspect_mean_30', 'raw_melspect_mean_31', 'raw_melspect_mean_32', 'raw_melspect_mean_33', 'raw_melspect_mean_34', 'raw_melspect_mean_35', 'raw_melspect_mean_36', 'raw_melspect_mean_37', 'raw_melspect_mean_38', 'raw_melspect_mean_39', 'raw_melspect_mean_40', 'raw_melspect_mean_41', 'raw_melspect_mean_42', 'raw_melspect_mean_43', 'raw_melspect_mean_44', 'raw_melspect_mean_45', 'raw_melspect_mean_46', 'raw_melspect_mean_47', 'raw_melspect_mean_48', 'raw_melspect_mean_49', 'raw_melspect_mean_50', 'raw_melspect_mean_51', 'raw_melspect_mean_52', 'raw_melspect_mean_53', 'raw_melspect_mean_54', 'raw_melspect_mean_55', 'raw_melspect_mean_56', 'raw_melspect_mean_57', 'raw_melspect_mean_58', 'raw_melspect_mean_59', 'cln_melspect_mean_0', 'cln_melspect_mean_1', 'cln_melspect_mean_2', 'cln_melspect_mean_3', 'cln_melspect_mean_4', 'cln_melspect_mean_5', 'cln_melspect_mean_6', 'cln_melspect_mean_7', 'cln_melspect_mean_8', 'cln_melspect_mean_9', 'cln_melspect_mean_10', 'cln_melspect_mean_11', 'cln_melspect_mean_12', 'cln_melspect_mean_13', 'cln_melspect_mean_14', 'cln_melspect_mean_15', 'cln_melspect_mean_16', 'cln_melspect_mean_17', 'cln_melspect_mean_18', 'cln_melspect_mean_19', 'cln_melspect_mean_20', 'cln_melspect_mean_21', 'cln_melspect_mean_22', 'cln_melspect_mean_23', 'cln_melspect_mean_24', 'cln_melspect_mean_25', 'cln_melspect_mean_26', 'cln_melspect_mean_27', 'cln_melspect_mean_28', 'cln_melspect_mean_29', 'cln_melspect_mean_30', 'cln_melspect_mean_31', 'cln_melspect_mean_32', 'cln_melspect_mean_33', 'cln_melspect_mean_34', 'cln_melspect_mean_35', 'cln_melspect_mean_36', 'cln_melspect_mean_37', 'cln_melspect_mean_38', 'cln_melspect_mean_39', 'cln_melspect_mean_40', 'cln_melspect_mean_41', 'cln_melspect_mean_42', 'cln_melspect_mean_43', 'cln_melspect_mean_44', 'cln_melspect_mean_45', 'cln_melspect_mean_46', 'cln_melspect_mean_47', 'cln_melspect_mean_48', 'cln_melspect_mean_49', 'cln_melspect_mean_50', 'cln_melspect_mean_51', 'cln_melspect_mean_52', 'cln_melspect_mean_53', 'cln_melspect_mean_54', 'cln_melspect_mean_55', 'cln_melspect_mean_56', 'cln_melspect_mean_57', 'cln_melspect_mean_58', 'cln_melspect_mean_59', 'raw_melspect_std_0', 'raw_melspect_std_1', 'raw_melspect_std_2', 'raw_melspect_std_3', 'raw_melspect_std_4', 'raw_melspect_std_5', 'raw_melspect_std_6', 'raw_melspect_std_7', 'raw_melspect_std_8', 'raw_melspect_std_9', 'raw_melspect_std_10', 'raw_melspect_std_11', 'raw_melspect_std_12', 'raw_melspect_std_13', 'raw_melspect_std_14', 'raw_melspect_std_15', 'raw_melspect_std_16', 'raw_melspect_std_17', 'raw_melspect_std_18', 'raw_melspect_std_19', 'raw_melspect_std_20', 'raw_melspect_std_21', 'raw_melspect_std_22', 'raw_melspect_std_23', 'raw_melspect_std_24', 'raw_melspect_std_25', 'raw_melspect_std_26', 'raw_melspect_std_27', 'raw_melspect_std_28', 'raw_melspect_std_29', 'raw_melspect_std_30', 'raw_melspect_std_31', 'raw_melspect_std_32', 'raw_melspect_std_33', 'raw_melspect_std_34', 'raw_melspect_std_35', 'raw_melspect_std_36', 'raw_melspect_std_37', 'raw_melspect_std_38', 'raw_melspect_std_39', 'raw_melspect_std_40', 'raw_melspect_std_41', 'raw_melspect_std_42', 'raw_melspect_std_43', 'raw_melspect_std_44', 'raw_melspect_std_45', 'raw_melspect_std_46', 'raw_melspect_std_47', 'raw_melspect_std_48', 'raw_melspect_std_49', 'raw_melspect_std_50', 'raw_melspect_std_51', 'raw_melspect_std_52', 'raw_melspect_std_53', 'raw_melspect_std_54', 'raw_melspect_std_55', 'raw_melspect_std_56', 'raw_melspect_std_57', 'raw_melspect_std_58', 'raw_melspect_std_59', 'raw_mfcc_mean_0', 'raw_mfcc_mean_1', 'raw_mfcc_mean_2', 'raw_mfcc_mean_3', 'raw_mfcc_mean_4', 'raw_mfcc_mean_5', 'raw_mfcc_mean_6', 'raw_mfcc_mean_7', 'raw_mfcc_mean_8', 'raw_mfcc_mean_9', 'raw_mfcc_mean_10', 'raw_mfcc_mean_11', 'raw_mfcc_mean_12', 'raw_mfcc_mean_13', 'raw_mfcc_mean_14', 'raw_mfcc_mean_15', 'raw_mfcc_mean_16', 'raw_mfcc_mean_17', 'raw_mfcc_mean_18', 'raw_mfcc_mean_19', 'raw_mfcc_std_0', 'raw_mfcc_std_1', 'raw_mfcc_std_2', 'raw_mfcc_std_3', 'raw_mfcc_std_4', 'raw_mfcc_std_5', 'raw_mfcc_std_6', 'raw_mfcc_std_7', 'raw_mfcc_std_8', 'raw_mfcc_std_9', 'raw_mfcc_std_10', 'raw_mfcc_std_11', 'raw_mfcc_std_12', 'raw_mfcc_std_13', 'raw_mfcc_std_14', 'raw_mfcc_std_15', 'raw_mfcc_std_16', 'raw_mfcc_std_17', 'raw_mfcc_std_18', 'raw_mfcc_std_19', 'yin_0', 'yin_1', 'yin_2', 'yin_3', 'yin_4', 'yin_5', 'yin_6', 'yin_7', 'yin_8', 'yin_9', 'yin_10', 'yin_11', 'yin_12', 'yin_13', 'raw_contrast_mean_0', 'raw_contrast_mean_1', 'raw_contrast_mean_2', 'raw_contrast_mean_3', 'raw_contrast_mean_4', 'raw_contrast_mean_5', 'raw_contrast_mean_6'])
        else:
            self.X = self.X.reindex(columns=['raw_melspect_mean_0', 'raw_melspect_mean_1', 'raw_melspect_mean_2', 'raw_melspect_mean_3', 'raw_melspect_mean_4', 'raw_melspect_mean_5', 'raw_melspect_mean_6', 'raw_melspect_mean_7', 'raw_melspect_mean_8', 'raw_melspect_mean_9', 'raw_melspect_mean_10', 'raw_melspect_mean_11', 'raw_melspect_mean_12', 'raw_melspect_mean_13', 'raw_melspect_mean_14', 'cln_melspect_mean_0', 'cln_melspect_mean_1', 'cln_melspect_mean_2', 'cln_melspect_mean_3', 'cln_melspect_mean_4', 'cln_melspect_mean_5', 'cln_melspect_mean_6', 'cln_melspect_mean_7', 'cln_melspect_mean_8', 'cln_melspect_mean_9', 'cln_melspect_mean_10', 'cln_melspect_mean_11', 'cln_melspect_mean_12', 'cln_melspect_mean_13', 'cln_melspect_mean_14', 'raw_mfcc_mean_0', 'raw_mfcc_mean_1', 'raw_mfcc_mean_2', 'raw_mfcc_mean_3', 'raw_mfcc_mean_4', 'raw_mfcc_std_0', 'raw_mfcc_std_1', 'raw_mfcc_std_2', 'raw_mfcc_std_3', 'raw_mfcc_std_4'])
        # self.X = self.X.reindex(columns=["raw_melspect_mean_0", "raw_melspect_mean_1", "raw_melspect_mean_2", "raw_melspect_mean_3", "raw_melspect_mean_4", "raw_melspect_mean_5", "raw_melspect_mean_6", "raw_melspect_mean_7", "raw_melspect_mean_8", "raw_melspect_mean_9", "raw_melspect_mean_10", "raw_melspect_mean_11", "raw_melspect_mean_12", "raw_melspect_mean_13", "raw_melspect_mean_14", "raw_melspect_mean_15", "raw_melspect_mean_16", "raw_melspect_mean_17", "raw_melspect_mean_18", "raw_melspect_mean_19", "raw_melspect_mean_20", "raw_melspect_mean_21", "raw_melspect_mean_22", "raw_melspect_mean_23", "raw_melspect_mean_24", "raw_melspect_mean_25", "raw_melspect_mean_26", "raw_melspect_mean_27", "raw_melspect_mean_28", "raw_melspect_mean_29", "raw_melspect_mean_30", "raw_melspect_mean_31", "raw_melspect_mean_32", "raw_melspect_mean_33", "raw_melspect_mean_34", "raw_melspect_mean_35", "raw_melspect_mean_36", "raw_melspect_mean_37", "raw_melspect_mean_38", "raw_melspect_mean_39", "raw_melspect_mean_40", "raw_melspect_mean_41", "raw_melspect_mean_42", "raw_melspect_mean_43", "raw_melspect_mean_44", "raw_melspect_mean_45", "raw_melspect_mean_46", "raw_melspect_mean_47", "raw_melspect_mean_48", "raw_melspect_mean_49", "raw_melspect_mean_50", "raw_melspect_mean_51", "raw_melspect_mean_52", "raw_melspect_mean_53", "raw_melspect_mean_54", "raw_melspect_mean_55", "raw_melspect_mean_56", "raw_melspect_mean_57", "raw_melspect_mean_58", "raw_melspect_mean_59", "cln_melspect_mean_0", "cln_melspect_mean_1", "cln_melspect_mean_2", "cln_melspect_mean_3", "cln_melspect_mean_4", "cln_melspect_mean_5", "cln_melspect_mean_6", "cln_melspect_mean_7", "cln_melspect_mean_8", "cln_melspect_mean_9", "cln_melspect_mean_10", "cln_melspect_mean_11", "cln_melspect_mean_12", "cln_melspect_mean_13", "cln_melspect_mean_14", "cln_melspect_mean_15", "cln_melspect_mean_16", "cln_melspect_mean_17", "cln_melspect_mean_18", "cln_melspect_mean_19", "cln_melspect_mean_20", "cln_melspect_mean_21", "cln_melspect_mean_22", "cln_melspect_mean_23", "cln_melspect_mean_24", "cln_melspect_mean_25", "cln_melspect_mean_26", "cln_melspect_mean_27", "cln_melspect_mean_28", "cln_melspect_mean_29", "cln_melspect_mean_30", "cln_melspect_mean_31", "cln_melspect_mean_32", "cln_melspect_mean_33", "cln_melspect_mean_34", "cln_melspect_mean_35", "cln_melspect_mean_36", "cln_melspect_mean_37", "cln_melspect_mean_38", "cln_melspect_mean_39", "cln_melspect_mean_40", "cln_melspect_mean_41", "cln_melspect_mean_42", "cln_melspect_mean_43", "cln_melspect_mean_44", "cln_melspect_mean_45", "cln_melspect_mean_46", "cln_melspect_mean_47", "cln_melspect_mean_48", "cln_melspect_mean_49", "cln_melspect_mean_50", "cln_melspect_mean_51", "cln_melspect_mean_52", "cln_melspect_mean_53", "cln_melspect_mean_54", "cln_melspect_mean_55", "cln_melspect_mean_56", "cln_melspect_mean_57", "cln_melspect_mean_58", "cln_melspect_mean_59", "raw_mfcc_mean_0", "raw_mfcc_mean_1", "raw_mfcc_mean_2", "raw_mfcc_mean_3", "raw_mfcc_mean_4", "raw_mfcc_mean_5", "raw_mfcc_mean_6", "raw_mfcc_mean_7", "raw_mfcc_mean_8", "raw_mfcc_mean_9", "raw_mfcc_mean_10", "raw_mfcc_mean_11", "raw_mfcc_mean_12", "raw_mfcc_mean_13", "raw_mfcc_mean_14", "raw_mfcc_mean_15", "raw_mfcc_mean_16", "raw_mfcc_mean_17", "raw_mfcc_mean_18", "raw_mfcc_mean_19", "raw_contrast_mean_0", "raw_contrast_mean_1", "raw_contrast_mean_2", "raw_contrast_mean_3", "raw_contrast_mean_4", "raw_contrast_mean_5", "raw_contrast_mean_6"])
        print(self.X.iloc[:5,:5].values)
        print(self.X.columns.tolist())

    def load_data(self, from_files):
        """Load Dataset:
            if from_files = True:   Return  self.df             -> Dataframe with label_paths and input_paths
            else:                   Return  self.X              -> DataFrame with 548 Features
                                            self.Y_raw          -> DataFrame with [Absolute Frequencies]
                                            self.Y_normalized   -> Daraframe with [Relaive Frequencies]"""
        # Dictionaries of Labels [Based on class_names.txt]
        self.labels_ids = pd.read_csv(os.path.join(self.data_dir, 'class_names.txt'), names=['id', 'label'],
                                      delimiter=': ', engine='python').to_dict()['label']
        self.labels_names = {v: k for k, v in self.labels_ids.items()}
        # Get ALL labels file names from data_dir
        self.label_files = glob(f'{self.data_dir}\**\*.labels.npy', recursive=True)
        # Get Corresponding Input file names from data_dir [Ignore Files That has no labels]
        self.data_files = [f_path.replace('.labels', '') for f_path in self.label_files]
        # TODO: Confirm input files Existence
        if from_files:
            self.load_file_names()
        else:
            self.load_labels_from_csv()
            self.load_data_from_csv()

    def get_label_by_id(self, idx):
        # Returns Label Name by ID
        return self.labels_ids[idx]

    def get_label_by_name(self, name):
        # Returns Label ID by Name
        return self.labels_names[name]

    def load_file_names(self):
        # Create DataFrame for Labels_Path and Input Path
        self.df_files = pd.DataFrame({'label_path': self.label_files, 'input_path': self.data_files})

    def load_labels_from_csv(self):
        # TODO: Method doc
        # Load / Create Labels Dataset
        try:
            # Load Labels Datasets [Y_raw, Y_normalized]
            self.Y_raw = pd.read_csv(os.path.join(self.data_dir, 'Y_raw.csv'))
            self.Y_normalized = pd.read_csv(os.path.join(self.data_dir, 'Y_normalized.csv'))
            # print('Labels Loaded!')
        except FileNotFoundError:
            # Create Labels Datasets [Y_raw, Y_Normalized]
            print('Extracting Labels from Files..')
            # Raw/Normalized Labels
            self.Y_raw = pd.DataFrame(columns=self.labels_ids)
            self.Y_normalized = pd.DataFrame(columns=self.labels_ids)
            # Crawl Files / Get Labels
            for f_path in tqdm(self.label_files):
                raw = pd.DataFrame(np.load(f_path))
                self.Y_raw, self.Y_normalized = self.normalize_labels(raw, self.Y_raw, self.Y_normalized)
            self.Y_raw.fillna(0, inplace=True)
            self.Y_normalized.fillna(0, inplace=True)
            # Save Files [For Future Use - Faster Loading]
            self.Y_raw.to_csv(os.path.join(self.data_dir, 'Y_raw.csv'), index=False)
            self.Y_normalized.to_csv(os.path.join(self.data_dir, 'Y_normalized.csv'), index=False)
        except:
            raise NotImplementedError('Something went wrong here!!')

    def load_data_from_csv(self):
        try:
            # Load Inputs Dataset [self.X]
            self.X = pd.read_csv(os.path.join(self.data_dir, 'X.csv'))
            # print('Data Loaded!')
        except FileNotFoundError:
            # Create Inputs Dataset [self.X]
            print('Extracting Features from files..')
            # Get Feature Names
            with open(os.path.join(self.data_dir, 'feature_names.txt'), 'r+') as f:
                feature_names = f.read().splitlines()
            self.X = pd.DataFrame(columns=feature_names)
            # Crawl Files / Get Data
            for f_path in tqdm(self.data_files):
                raw = pd.DataFrame(np.load(f_path), columns=feature_names)
                self.X = pd.concat([self.X, raw], ignore_index=True, sort=False)
            # Save Data for Future Use
            self.X.to_csv(os.path.join(self.data_dir, 'X.csv'), index=False)
        except:
            raise NotImplementedError('Something went wrong here!')

    def load_data_infer(self):
        try:
            self.X = pd.read_csv(os.path.join(self.data_dir, 'X_infer__.csv'))
        except FileNotFoundError:
            self.data_files = glob(f"{os.path.join(f'{self.data_dir}', 'testset')}/*.npy")
            with open(os.path.join(self.data_dir, 'feature_names.txt'), 'r+') as f:
                feature_names = f.read().splitlines()
            self.X = pd.DataFrame(columns=feature_names)
            for f_path in tqdm(self.data_files):
                raw = pd.DataFrame(np.load(f_path), columns=feature_names)
                self.X = pd.concat([self.X, raw], ignore_index=True, sort=False)
            self.X.to_csv(os.path.join(self.data_dir, 'X_infer.csv'), index=False)
        except:
            raise NotImplementedError('Inference load_data_infer!!')

    def load_feature_sets(self):
        """Load Features Sets
                Returns:        self.feature_sets   -> Array of Feature Names
                                self.features       -> Dictionary feature: [feature columns]"""
        features = self.X.columns.tolist()
        self.feature_sets = ['raw_melspect_mean', 'raw_melspect_std', 'cln_melspect_mean', 'cln_melspect_std',
                             'raw_mfcc_mean', 'raw_mfcc_std', 'cln_mfcc_mean', 'cln_mfcc_std',
                             'raw_mfcc_d_mean', 'raw_mfcc_d_std', 'cln_mfcc_d_mean', 'cln_mfcc_d_std',
                             'raw_mfcc_d2_mean', 'raw_mfcc_d2_std', 'cln_mfcc_d2_mean', 'cln_mfcc_d2_std',
                             'zcr', 'yin', 'bandwidth_mean', 'bandwidth_std',
                             'flatness_mean', 'flatness_std', 'centroid_mean', 'centroid_std',
                             'flux_mean', 'flux_std', 'energy_mean', 'energy_std', 'power_mean', 'power_std',
                             'raw_contrast_mean', 'raw_contrast_std', 'cln_contrast_mean', 'cln_contrast_std',
                             ]
        for feature_set in self.feature_sets:
            self.features[feature_set] = [x for x in features if feature_set in x]

    def reorder_cols(self):
        with open('models_features.json', 'r') as f:
            features = json.load(f)
        order = features[model_name]['order']
        self.X = self.X.reindex(columns=order)

    def analyze_dataset(self):
        self.labels_analysis = self.analyze_labels()
        self.features_analysis = self.analyze_features()
        self.analyze_feature_labels()
        fs = ['power_mean', 'power_std']
        for f in fs:
            fig = px.line(self.X[self.features[f]])
            fig.update_yaxes(gridcolor='black')
            fig.update_layout(
                              template='plotly_white',
                              margin=dict(pad=7),
                              # font=dict(family="Helvetica", size=28),
                              plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                              legend=dict(orientation="h")
                              )
            fig.show()
        reports_dir = os.path.join(self.output_dir, 'reports')
        Path(reports_dir).mkdir(parents=True, exist_ok=True)

    def analyze_labels(self):
        # Stupid Bug
        # TODO Fix Above Index Type
        # TODO Use Concat Instead
        analysis_cols = ['name', 'distribution', 'mean', 'median', 'var', 'std',
                         'agrmnt_total', 'agrmnt_strong', 'agrmnt_medium', 'agrmnt_conflict',
                         'call_avg_r', 'call_avg_s', 'call_median']
        labels_analysis = pd.DataFrame(columns=analysis_cols)

        labels_analysis[['distribution', 'mean', 'median', 'var', 'std']] = self.analyze_df_basic(self.Y_normalized)
        labels_analysis.index = labels_analysis.index.astype(int)
        labels_analysis[['agrmnt_total', 'agrmnt_strong', 'agrmnt_medium', 'agrmnt_conflict']] = \
            self.analyze_labels_agreement(self.Y_normalized, self.labels_ids)
        labels_analysis.index = labels_analysis.index.astype(int)
        labels_analysis[['call_avg_r', 'call_avg_s', 'call_median']], avg_calls_sec = \
            self.analyze_call_duration(self.Y_normalized, self.labels_ids)
        labels_analysis['name'] = list(self.labels_names)

        # Visualize
        if self.plot:
            self.plot_basic(labels_analysis, 'mean', 'var',
                            title='Basic Labels Analysis', bar_name='Distribution %', xaxis='Labels')
            self.plot_avg_call(avg_calls_sec, labels_analysis)
            self.plot_agrmnt(labels_analysis)
        return labels_analysis

    def analyze_features(self):
        analysis_cols = ['name', 'mean', 'median', 'var', 'std',
                         'top_15_exp_var_sums', 'avg_inter_corr']
        features_analysis = pd.DataFrame(columns=analysis_cols)

        features_analysis['name'] = list(self.features)
        features_analysis.index = features_analysis.index.astype(int)
        features_analysis[['mean', 'median', 'var', 'std']] = self.analyze_feature_sets_basic(self.X, self.features)
        features_analysis[['avg_inter_corr']], inter_corr_mats = self.analyze_features_inter_corr(self.X, self.features)
        features_analysis['top_15_exp_var_sums'], top_inter_features = \
            self.analyze_features_inter_exp_variance(self.X, self.features)

        # Use Reasonably Inter-Correlated Features [Doesn't make sense to compare low correlated features]
        highly_correlated_features = \
            features_analysis.loc[features_analysis['avg_inter_corr'].abs() >= 0.4, 'name'].values
        intra_features_analysis, intra_features_mats = \
            self.analyze_features_intra_corr(self.X, self.X.columns.tolist(), highly_correlated_features)

        # Visuals
        if self.plot:
            f_names = features_analysis['name']
            features_analysis['name'] = f_names.index
            self.plot_basic(features_analysis, 'mean', 'var',
                            title='Basic Features Analysis', xaxis='Feature Sets',
                            log_y=True, xaxis_orient=-90)
            self.plot_inter_corr(features_analysis, inter_corr_mats, highly_correlated_features)
            self.plot_intra_corr(intra_features_analysis, intra_features_mats)
        return features_analysis

    def analyze_feature_labels(self):
        _vars = []
        for label in tqdm(self.labels_ids):
            for feature_set in self.feature_sets:
                pca = PCA(n_components=1)
                pca.fit(self.X.loc[self.Y_normalized[str(label)] > 0, self.features[feature_set]])
                _vars.append([self.get_label_by_id(label), feature_set, *pca.explained_variance_ratio_])
        _vars = pd.DataFrame(_vars, columns=['Labels', 'Features', 'avg_var']).pivot_table(columns='Features',
                                                                                           index='Labels',
                                                                                           values='avg_var')
        if self.plot:
            fig = px.imshow(_vars, text_auto=False)
            fig.update_xaxes(tickangle=-90)
            fig.update_layout(
                xaxis_title="Feature Sets", yaxis_title="Labels",
                template='plotly_white',
                margin=dict(pad=7),
                # font=dict(family="Helvetica", size=28),
                plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                legend=dict(orientation="h")
            )
            fig.show()

            fig = px.imshow(_vars[['raw_melspect_std', 'cln_melspect_std']], text_auto=False)
            fig.update_xaxes(tickangle=-90)
            fig.update_layout(
                xaxis_title="Feature Sets", yaxis_title="Labels",
                template='plotly_white',
                margin=dict(pad=7),
                # font=dict(family="Helvetica", size=28),
                plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                legend=dict(orientation="h")
            )
            fig.show()

            fig = px.imshow(_vars[['raw_mfcc_d_std', 'raw_mfcc_d2_std', 'cln_mfcc_d_std', 'cln_mfcc_d2_std']],
                            text_auto=False)
            fig.update_xaxes(tickangle=-90)
            fig.update_layout(
                xaxis_title="Feature Sets", yaxis_title="Labels",
                template='plotly_white',
                margin=dict(pad=7),
                # font=dict(family="Helvetica", size=28),
                plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                legend=dict(orientation="h")
            )
            fig.show()

            fig = px.imshow(_vars[['raw_mfcc_std', 'cln_mfcc_std']], text_auto=False)
            fig.update_xaxes(tickangle=-90)
            fig.update_layout(title="Explained Variance for each PCA-Reduced Feature Set",
                              xaxis_title="Feature Sets", yaxis_title="Labels",
                              template='plotly_white',
                              margin=dict(pad=7),
                              # font=dict(family="Helvetica", size=28),
                              plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                              legend=dict(orientation="h")
                              )
            fig.show()

            fig = px.imshow(_vars[['raw_contrast_std', 'cln_contrast_std']], text_auto=False)
            fig.update_xaxes(tickangle=-90)
            fig.update_layout(title="Explained Variance for each PCA-Reduced Feature Set",
                              xaxis_title="Feature Sets", yaxis_title="Labels",
                              template='plotly_white',
                              margin=dict(pad=7),
                              # font=dict(family="Helvetica", size=28),
                              plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                              legend=dict(orientation="h")
                              )
            fig.show()

    @staticmethod
    def analyze_df_basic(df, cols=None):
        # Basic Labels Analysis
        if cols is None:
            cols = ['distribution', 'mean', 'median', 'var', 'std']
        analysis = pd.DataFrame(columns=cols)
        if 'distribution' in cols:
            analysis['distribution'] = df.sum(axis=0)
        analysis['mean'] = df.mean()
        analysis['median'] = df.median()
        analysis['var'] = df.var()
        analysis['std'] = df.std()
        return analysis

    @staticmethod
    def analyze_labels_agreement(df, labels_ids, cols=None):
        # Annotations Analysis
        if cols is None:
            cols = ['agrmnt_total', 'agrmnt_strong', 'agrmnt_medium', 'agrmnt_conflict']
        analysis = pd.DataFrame(columns=cols)
        total_agreement, strong_agreement, medium_agreement, conflict = [], [], [], []
        for label_id, label_name in enumerate(labels_ids):
            _id = str(label_id)
            # Ignore Zeros [Only Labeled Items]
            non_zeros = df[df[_id] != 0]
            # Calculate Bins
            total_agreement.append(100 * non_zeros.loc[(non_zeros[_id] == 1), _id].shape[0] / len(non_zeros))
            strong_agreement.append(
                100 * non_zeros.loc[(non_zeros[_id] < 1) & (non_zeros[_id] >= 0.75), _id].shape[0] / len(non_zeros))
            medium_agreement.append(
                100 * non_zeros.loc[(non_zeros[_id] < 0.75) & (non_zeros[_id] >= 0.5), _id].shape[0] / len(non_zeros))
            conflict.append(
                100 * non_zeros.loc[(non_zeros[_id] < 0.5) & (non_zeros[_id] > 0), _id].shape[0] / len(non_zeros))
        analysis['agrmnt_total'] = total_agreement
        analysis['agrmnt_strong'] = strong_agreement
        analysis['agrmnt_medium'] = medium_agreement
        analysis['agrmnt_conflict'] = conflict
        return analysis

    @staticmethod
    def analyze_call_duration(df, labels_ids, cols=None):
        # Average Call Duration
        if cols is None:
            cols = ['call_avg_r', 'call_avg_s', 'call_median']
        analysis = pd.DataFrame(columns=cols)
        avg_calls_rows, avg_calls_seconds, median_calls, total_lengths = [], [], [], []
        for label_id, label_name in enumerate(labels_ids):
            # Select All NonZeros
            non_zeros = df[df[str(label_id)] != 0].index.values
            # Split NonZeros to arrays of consecutive calls
            splits = np.split(non_zeros, np.where(np.diff(non_zeros) != 1)[0] + 1)
            # Check Lengths of each Call
            lengths = np.array([len(x) for x in splits])
            # Calculate Lengths
            total_lengths.append(np.array(lengths * 0.2))
            avg_calls_rows.append(lengths.mean())
            avg_calls_seconds.append(lengths.mean() * 0.2)
            median_calls.append(np.median(lengths))
        analysis['call_avg_r'] = np.array(avg_calls_rows)
        analysis['call_avg_s'] = np.array(avg_calls_seconds)
        analysis['call_median'] = np.array(median_calls)
        max_num_lengths = sorted(len(x) for x in total_lengths)
        total_lengths = [np.pad(x, pad_width=(max_num_lengths[-1] - len(x)), mode='constant', constant_values=0)[
                         max_num_lengths[-1] - len(x):] for x in total_lengths]
        return analysis, pd.DataFrame(np.array(total_lengths).transpose())

    @staticmethod
    def analyze_feature_sets_basic(df, features, cols=None):
        if cols is None:
            cols = ['mean', 'median', 'var', 'std', ]
        analysis = pd.DataFrame(columns=cols)
        i = 0
        for feature_set in tqdm(features):
            analysis.loc[i, 'mean'] = df[features[feature_set]].stack().mean()
            analysis.loc[i, 'median'] = df[features[feature_set]].stack().median()
            analysis.loc[i, 'var'] = df[features[feature_set]].stack().var()
            analysis.loc[i, 'std'] = df[features[feature_set]].stack().std()
            i += 1
        return analysis

    @staticmethod
    def analyze_features_inter_corr(df, features, cols=None):
        if cols is None:
            cols = ['avg_inter_corr']
        avg_inter_corrs, std_inter_corrs = [], []
        corrs = {}
        analysis = pd.DataFrame(columns=cols)
        for feature_set in features:
            corr = df[features[feature_set]].corr()
            corrs[feature_set] = corr
            avg_corr = np.arctanh(corr.replace(1, 0).values).mean()
            avg_inter_corrs.append(avg_corr.mean())
        analysis['avg_inter_corr'] = avg_inter_corrs
        return analysis, corrs

    def analyze_features_inter_exp_variance(self, df, features, cols=None):
        if cols is None:
            cols = ['top_15_exp_var_sums']
        top_15_inter_features = {}
        top_15_exp_vars, top_15_exp_var_sums = [], []
        analysis = pd.DataFrame(columns=cols)
        for feature_set in features:
            top_15_var, top_15_ind = self.get_exp_variance(df[features[feature_set]])
            top_15_exp_vars.append(top_15_var)
            top_15_exp_var_sums.append(top_15_var[:15].sum())
            top_15_inter_features[feature_set] = [features[feature_set][i] for i in top_15_ind[:15]]
        analysis['top_15_exp_var_sums'] = top_15_exp_var_sums
        return analysis, top_15_inter_features

    @staticmethod
    def analyze_features_intra_corr(df, all_features, features):
        # Create combinations for features
        groups = list(combinations(features, 2))
        # all_features = self.X.columns.tolist()
        highly_corr_feature_sets, highly_corr_feature_mats = [], {}
        for grp in tqdm(groups):
            # Feature Names
            gp1 = [x for x in all_features if grp[0] in x]
            gp2 = [x for x in all_features if grp[1] in x]
            df_corr = df[[*gp1, *gp2]].corr()
            # Average Correlation for each Feature set
            mean_corr_gp1 = np.arctanh(df_corr.iloc[:len(gp1), :len(gp1)].replace(1, 0).values).mean()
            mean_corr_gp2 = np.arctanh(df_corr.iloc[:len(gp1), len(gp1):].replace(1, 0).values).mean()
            # Difference between Average Correlations
            avg_corr_diff = abs(abs(mean_corr_gp1) - abs(mean_corr_gp2))
            # highly_corr_feature_sets.append([*grp, avg_corr_diff])
            highly_corr_feature_sets.append([f'{grp[0]} vs. {grp[1]}', *grp, avg_corr_diff])
            if avg_corr_diff < 0.1:
                # highly_corr_feature_sets.append([*grp, avg_corr_diff])
                highly_corr_feature_mats[f'{grp[0]} vs. {grp[1]}'] = df_corr
        # analysis = pd.DataFrame(np.array(highly_corr_feature_sets), columns=['fs1', 'fs2', 'avg_corr_diff'])
        analysis = pd.DataFrame(np.array(highly_corr_feature_sets), columns=['group', 'fs1', 'fs2', 'avg_corr_diff'])
        # analysis = analysis.pivot_table(columns='fs1', index='fs2', values='avg_corr_diff')
        return analysis, highly_corr_feature_mats

    @staticmethod
    def get_exp_variance(df):
        x_std = StandardScaler().fit_transform(df)
        cov_matrix = np.cov(x_std.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        eig_tuple = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
        eig_tuple.sort(reverse=True)

        sort_ind = np.argsort(eigenvalues)[::-1]
        sort_eig = eigenvalues[sort_ind]
        return np.array([eigenvalue / np.sum(eigenvalues) for eigenvalue in sort_eig]), sort_ind

    @staticmethod
    def plot_basic(labels_analysis, bar, line, title='',
                   bar_name='Mean', line_name='Variance', xaxis='', yaxis='Values', xaxis_orient=0,
                   log_x=False, log_y=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=labels_analysis['name'], y=labels_analysis[line],
                                 name=line_name, mode='lines', line_shape='spline', ))
        fig.add_trace(go.Bar(x=labels_analysis['name'], y=labels_analysis[bar], name=bar_name,
                             width=[0.2 for _ in labels_analysis.index]))
        fig.update_layout(title=title,
                          xaxis_title=xaxis, yaxis_title=yaxis,
                          template='plotly_white',
                          margin=dict(pad=7 if xaxis_orient else 5),
                          font=dict(family="Helvetica", size=28),
                          plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                          # legend=dict(yanchor="top", y=0.42, xanchor="right")
                          legend=dict(orientation="h")
                          )
        fig.update_yaxes(rangemode="tozero", type="log" if log_y else "linear", gridwidth=1, gridcolor='black')
        fig.update_xaxes(showgrid=True, gridwidth=2, type="log" if log_x else "category", tickangle=xaxis_orient)

        fig.show()

    @staticmethod
    def plot_multiple_heatmaps(intra_features_mats, features=None, n_rows=4, n_cols=5, title='Correlation Matrix'):
        if not isinstance(features, list):
            features = list(intra_features_mats)
        fig = make_subplots(rows=n_rows, cols=n_cols,
                            print_grid=False,
                            subplot_titles=features)
        i, j = 0, 0
        for grp in features:
            fig.add_trace(go.Heatmap(z=intra_features_mats[grp].values.tolist(),
                                     colorscale='brwnyl', coloraxis="coloraxis",
                                     zmin=-1, zmid=0, zmax=1),
                          row=j + 1, col=i + 1)
            if i < len(features) // 4:
                i += 1
            elif i == len(features) // 4:
                j += 1
                i = 0
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_annotations(font=dict(family="Helvetica", size=14))
        fig.update_layout(title=title, coloraxis_cmin=-1, coloraxis_cmax=1,
                          font=dict(family="Helvetica", size=28),
                          plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        fig.show()

    def plot_avg_call(self, avg_calls_sec, labels_analysis):
        avg_calls_sec = avg_calls_sec.rename(columns=self.labels_ids)
        fig = px.box(avg_calls_sec.replace({0: np.nan}), y=avg_calls_sec.columns,
                     log_y=True, points='all', notched=True,
                     title="")
        fig.update_traces(quartilemethod="exclusive")
        fig.add_trace(go.Scatter(x=avg_calls_sec.columns, y=labels_analysis['call_avg_s'],
                                 name='Mean/Class', mode='lines', line_shape='spline'))
        fig.add_hline(y=labels_analysis['call_avg_s'].median(), line_width=3, line_dash="dash",
                      name='Median of all classes')
        fig.update_layout(title="Average bird call durations in seconds",
                          xaxis_title="Bird Classes", yaxis_title="Call Duration (s)",
                          template='plotly_white',
                          font=dict(family="Helvetica", size=28),
                          plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                          legend=dict(orientation="h")
                          )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
        fig.show()

    def plot_agrmnt(self, labels_analysis):
        fig = go.Figure()
        columns = ['agrmnt_total', 'agrmnt_strong', 'agrmnt_medium', 'agrmnt_conflict']
        # colors = ['rgb(102, 197, 204)', 'rgb(135, 197, 95)', 'rgb(246, 207, 113)', 'rgb(254, 136, 177)']
        colors = ['rgb(47, 138, 196)', 'rgb(36, 121, 108)', 'rgb(229, 134, 6)', 'rgb(102, 17, 0)']
        for k, v in enumerate(['Total', 'Strong', 'Medium', 'Conflict']):
            data = labels_analysis[columns[k]]
            y_names = [self.labels_ids[x] for x in labels_analysis.index]
            fig.add_trace(go.Bar(x=data, y=y_names, name=v,
                                 orientation='h', marker_color=colors[k]))
        fig.update_layout(title="Annotators Agreement Percentage", barmode='stack',
                          xaxis_title="Percentage", yaxis_title="Bird classes",
                          template='plotly_white',
                          font=dict(family="Helvetica", size=28),
                          plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)',
                          legend=dict(orientation="h")
                          )
        fig.show()

    def plot_inter_corr(self, features_analysis, inter_corr_mats, highly_correlated_features):
        fig = px.scatter(features_analysis, x='name', y='avg_inter_corr')
        fig.add_hline(y=0)
        fig.add_hline(y=0.4, line_width=2, line_dash="dash", )
        fig.update_layout(title='Average Inter-Correlation between feature sets',
                          xaxis_title='Labels', yaxis_title='Avg. Correlation Coefficient',
                          template='plotly_white',
                          margin=dict(pad=7),
                          font=dict(family="Helvetica", size=28),
                          plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)'
                          )
        fig.update_yaxes(rangemode="tozero", gridwidth=1)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black', type='category')
        fig.update_traces(marker=dict(size=20), selector=dict(mode='markers'))
        fig.show()
        self.plot_multiple_heatmaps(inter_corr_mats, features=highly_correlated_features.tolist(),
                                    n_rows=4, n_cols=3, title='Top Inter Correlated Feature sets Matrices')

    def plot_intra_corr(self, intra_features_analysis, intra_features_mats):
        # fig = px.scatter(intra_features_analysis.sort_values(by='avg_corr_diff'),
        fig = px.scatter(intra_features_analysis,
                         x=intra_features_analysis.index, y='avg_corr_diff',
                         log_y=True)
        fig.add_hline(y=0.1, line_width=2, line_dash="dash")
        fig.update_layout(title='Absolute Mean Difference between each pair of "Highly Correlated" feature sets',
                          xaxis_title='Group Pairs', yaxis_title='Absolute Mean Difference',
                          template='plotly_white',
                          margin=dict(pad=7),
                          font=dict(family="Helvetica", size=28),
                          plot_bgcolor='rgba(0, 0, 0, 0)', paper_bgcolor='rgba(0, 0, 0, 0)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black', type='category')
        fig.update_traces(marker=dict(size=20), selector=dict(mode='markers'))
        fig.show()

        self.plot_multiple_heatmaps(intra_features_mats, title='Top Intra Correlated Feature sets Matrices')

    def select_features(self, feature_sets):
        features = []
        for fs in feature_sets:
            features.extend(self.features[fs])
        return self.X[features]

    def balance_dataset(self):
        if not self.binary:
            # Drop Some of the 1s
            ind = []
            for c in self.Y_OneHot.columns:
                if c != '0':
                    ind.extend(self.Y_OneHot.loc[self.Y_OneHot[c] == 1, c].index.values.tolist())
                else:
                    ind.extend(
                        np.random.choice(self.Y_OneHot.loc[self.Y_OneHot[c] == 1, '0'].index.values, size=10000).tolist())
        else:
            ind = np.random.choice(self.Y_normalized.loc[self.Y_normalized['0'] == 1, '0'].index.values, size=28000).tolist()
            ind.extend(np.random.choice(self.Y_normalized.loc[self.Y_normalized['1'] == 1, '1'].index.values, size=28000).tolist())

        self.Y_single = self.Y_single[ind]
        self.Y_raw = self.Y_raw.iloc[ind, :]
        self.Y_normalized = self.Y_normalized.iloc[ind, :]
        self.Y_OneHot = self.Y_OneHot.iloc[ind, :]
        self.X = self.X.iloc[ind, :]

        # self.Y_single.reset_index(inplace=True)
        self.Y_raw.reset_index(inplace=True, drop=True)
        self.Y_normalized.reset_index(inplace=True, drop=True)
        self.Y_OneHot.reset_index(inplace=True, drop=True)
        self.X.reset_index(inplace=True, drop=True)

    def create_class_weights(self, percent=False, gain_weight=False):
        self.class_weights = []
        classes = self.Y_normalized.columns
        if self.binary:
            if percent:
                for c in classes:
                    self.class_weights.append((self.Y_normalized[c] >= 0.5).sum() / len(self.Y_normalized))
                    self.class_weights = np.array(self.class_weights)[::-1]
            else:
                for c in classes:
                    self.class_weights.append((self.Y_normalized[c] >= 0.5).sum())
                self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
                self.class_weights *= len(self.Y_normalized)
            self.weights = pd.DataFrame([0] * len(self.Y_normalized))
            for i, c in enumerate(classes):
                self.weights.loc[self.Y_normalized[c] >= 0.5] = self.class_weights[i]
            # self.class_weights = self.class_weights
        elif self.non_binary:
            if percent:
                for c in classes:
                    self.class_weights.append((self.Y_normalized[c] >= 0.5).sum() / len(self.Y_normalized))
                self.class_weights = np.array(self.class_weights)
                print(self.class_weights)
                self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
                self.class_weights /= self.class_weights.sum()
            else:
                for c in classes:
                    self.class_weights.append((self.Y_normalized[c] >= 0.5).sum())
                self.class_weights = max(self.class_weights) / np.array(self.class_weights)
                # self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
                # self.class_weights *= len(self.Y_normalized)
                # [0.11105017 0.15178107 0.13821373 0.03282143 0.05487558 0.04511864]
            self.weights = pd.DataFrame([0] * len(self.Y_normalized))
            for i, c in enumerate(classes):
                self.weights.loc[self.Y_normalized[c] >= 0.5] = self.class_weights[i]

            # self.class_weights[0] *= 1.2
            # self.class_weights[1] *= 0.6
            # self.class_weights[2] *= 0.7
            # self.class_weights[3] *= 7
            # self.class_weights[4] *= 1.4
            # self.class_weights[5] *= 1.8
        else:
            for c in classes:
                if percent:
                    self.class_weights.append((self.Y_normalized[c] >= 0.5).sum() / len(self.Y_normalized))
                else:
                    self.class_weights.append((self.Y_normalized[c] >= 0.5).sum())
            self.class_weights = np.array(self.class_weights)
            if not gain_weight:
                if percent:
                    self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
                    self.class_weights /= self.class_weights.sum()
                else:
                    # self.class_weights = len(self.Y_normalized) / self.class_weights
                    self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
                    self.class_weights *= len(self.Y_normalized)
                    self.class_weights[0] *= 1.5
                    self.weights = pd.DataFrame([0] * len(self.Y_normalized))
                    for i, c in enumerate(classes):
                        self.weights.loc[self.Y_normalized[c] >= 0.5] = self.class_weights[i]
            else:
                self.class_weights = np.array([1.0 / freq for freq in self.class_weights])
                gain_matrix = np.array([[0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
                                        [-0.25, 1, -0.3, -0.1, -0.1, -0.1, -0.1],
                                        [-0.02, -0.1, 1, -0.1, -0.1, -0.1, -0.1],
                                        [-0.25, -0.1, -0.3, 1, -0.1, -0.1, -0.1],
                                        [-0.25, -0.1, -0.3, -0.1, 1, -0.1, -0.1],
                                        [-0.25, -0.1, -0.3, -0.1, -0.1, 1, -0.1],
                                        [-0.25, -0.1, -0.3, -0.1, -0.1, -0.1, 1]])
                self.class_weights = np.matmul(self.class_weights, gain_matrix)
                if percent:
                    self.class_weights /= self.class_weights.sum()
        print(self.class_weights.sum(), self.class_weights)

    def __len__(self):
        if not self.mode == 'infer':
            if self.from_files:
                return len(self.df_files)
            else:
                return int(len(self.Y_normalized) - self.win_size)
        else:
            if self.from_files:
                return len(self.df_files)
            else:
                return len(self.X)

    def get_inputs_from_df(self, idx):
        if self.x_type == 'single' or self.x_type == '1d':
            return self.X.iloc[idx, :].values
        else:
            return self.X.iloc[idx - self.win_size:idx, :].values
            # return self.X.iloc[idx - int(self.win_size / 2):idx + int(self.win_size / 2), :].values

    def get_labels_from_df(self, idx):
        if self.x_type == 'single' or self.x_type == '1d':
            return self.Y_normalized.iloc[idx, :].values
            # if not self.binary:
            #     return self.Y_normalized.iloc[idx, :].values
            # else:
            #     return self.Y_single[idx]
        else:
            return self.Y_normalized.iloc[idx, :].values
            # return self.Y_normalized.iloc[idx - self.win_size:idx, :].values
            # if not self.binary:
            #     # return self.Y_normalized.iloc[idx - int(self.win_size / 2):idx + int(self.win_size / 2), :].values
            #     return self.Y_normalized.iloc[idx - self.win_size:idx, :].values
            # else:
            #     raise NotImplementedError('SHHHHHHHHHHHHHHIIIIIIIIIIIIIIIIIT in getting the label')

    def get_inputs_fom_file(self, idx):
        return np.load(self.df_files.loc[idx, 'input_path'])

    def get_labels_from_file(self, idx):
        raw = pd.DataFrame(np.load(self.df_files.loc[idx, 'label_path']))
        _, y_normalized = self.normalize_labels(raw)
        return y_normalized

    def normalize_labels(self, raw, Y_raw=None, Y_normalized=None):
        if not Y_raw:
            Y_raw = pd.DataFrame(columns=self.labels_ids)
        if not Y_normalized:
            Y_normalized = pd.DataFrame(columns=self.labels_ids)
        # Absolute Frequency
        abs_freq = raw.T.apply(pd.value_counts).T.fillna(0)
        Y_raw = pd.concat([Y_raw, abs_freq], ignore_index=True, sort=False)
        # Relative Frequency
        rel_freq = abs_freq.div(abs_freq.sum(axis=1), axis=0)
        Y_normalized = pd.concat([Y_normalized, rel_freq], ignore_index=True, sort=False)
        return Y_raw, Y_normalized

    def get_labels_mean(self, Y):
        other = 0 if self.from_files else str(0)
        return Y[Y[other] != 1].mean().fillna(0).values

    @staticmethod
    def get_labels_hot_enc(Y):
        y = np.zeros(shape=Y.shape)
        y[np.argmax(Y)] = 1.
        return y

    def create_onehot_enc(self):
        # TODO Decide Threshold [Very Important]
        self.Y_OneHot = self.Y_normalized.copy()
        for col in self.Y_OneHot.columns:
            if col != '0':
                self.Y_OneHot.loc[self.Y_OneHot[col] != 0, col] = 1
            else:
                self.Y_OneHot.loc[self.Y_OneHot[col] != 1, col] = 0
        return self.Y_OneHot.astype(int)

    def __getitem__(self, idx):
        if not self.from_files:
            if not self.mode == 'infer':
                x, y = self.get_inputs_from_df(idx), self.get_labels_from_df(idx)
                # Training / Validation
                # if (idx > self.win_size) and (idx < (len(self) - (self.win_size))):
                #     x, y = self.get_inputs_from_df(idx), self.get_labels_from_df(idx)
                # else:
                #     x, y = self.get_inputs_from_df(self.win_size), self.get_labels_from_df(self.win_size)
            else:
                # Inference
                x = self.get_inputs_from_df(idx)
        else:
            x, y = self.get_inputs_fom_file(idx), self.get_labels_from_file(idx)

        # print(x.shape, y.shape)
        # One Hot Encoding
        if not self.mode == 'infer':
            if self.x_type == 'single':
                # return torch.FloatTensor(x), torch.FloatTensor(self.get_labels_hot_enc(self.get_labels_mean(y)))
                # return torch.FloatTensor(x), torch.FloatTensor(self.get_labels_hot_enc(y))
                return torch.FloatTensor(x), torch.FloatTensor(y)
                # if not self.binary:
                #     return torch.FloatTensor(x), torch.FloatTensor(self.get_labels_hot_enc(y))
                # else:
                #     return torch.FloatTensor(x), torch.tensor(np.expand_dims(y, axis=0)).to(torch.float32)
            elif self.x_type == '1d' or self.x_type == '2d':
                # return torch.unsqueeze(torch.FloatTensor(x.expand_dims()), 0),  torch.FloatTensor(self.get_labels_hot_enc(self.get_labels_mean(y)))
                # return torch.FloatTensor(np.expand_dims(x, axis=0)),  torch.FloatTensor(self.get_labels_hot_enc(self.get_labels_mean(y)))
                # return torch.FloatTensor(np.expand_dims(x, axis=0)), torch.FloatTensor(self.get_labels_hot_enc(y))
                return torch.FloatTensor(np.expand_dims(x, axis=0)), torch.FloatTensor(y)
                # if not self.binary:
                #     return torch.FloatTensor(np.expand_dims(x, axis=0)),  torch.FloatTensor(self.get_labels_hot_enc(y))
                # else:
                #     return torch.FloatTensor(np.expand_dims(x, axis=0)),  torch.tensor(np.expand_dims(y, axis=0)).to(torch.float32)
            else:
                raise NotImplementedError('x_type not identified')
        else:
            if self.x_type == 'single':
                # print(x.min(), x.max())
                return torch.FloatTensor(x)
            elif self.x_type == '1d' or self.x_type == '2d':
                # return torch.unsqueeze(torch.FloatTensor(x.expand_dims()), 0),  torch.FloatTensor(self.get_labels_hot_enc(self.get_labels_mean(y)))
                return torch.FloatTensor(np.expand_dims(x, axis=0))
            else:
                raise NotImplementedError('x_type not identified')


class Loader:
    def __init__(self, data_dir='data', analyze=False, from_files=False, balance_dataset=False, mode='train',
                 cross_valid=False, kfolds=6, binary=False, non_binary=False, x_type='1d',
                 win_size=100, features='*', reduce_features=[], reducer='model',
                 train_size=0.8, test_size=0.1,
                 seed=951753, **kwargs):
        self.seed = seed
        self.features = features
        self.cross_valid, self.kfolds = cross_valid, kfolds
        self.binary, self.non_binary = binary, non_binary

        torch.manual_seed(seed)
        self.ds = AudioDS(data_dir=data_dir, analyze=analyze, from_files=from_files, win_size=win_size,
                          binary=binary, non_binary=non_binary,
                          features=features, reduce_features=reduce_features, reducer=reducer, balance_dataset=balance_dataset,
                          mode=mode, x_type=x_type)
        if mode == 'train':
            self.folds = None

            self.train_size = train_size
            self.test_size = test_size
            if self.test_size > 0.5:
                raise AttributeError('test_size Set Must be smaller than 50%')

            if not self.cross_valid:
                self.train_ds, self.valid_ds, self.test_ds = random_split(self.ds,
                                                                          [1 - (2 * test_size), test_size, test_size],
                                                                          generator=torch.Generator().manual_seed(seed)
                                                                          )
                # weights = torch.DoubleTensor(self.ds.weights.values.flatten())[self.train_ds.indices]
                # print(self.train_ds.indices)
                # _sampler = sampler.WeightedRandomSampler(weights, len(weights), replacement=True)

                # self.train = DataLoader(self.train_ds, sampler=_sampler, **kwargs)
                self.train = DataLoader(self.train_ds, **kwargs)
                self.valid = DataLoader(self.valid_ds, **kwargs)
                self.test = DataLoader(self.test_ds, **kwargs)
            else:
                t_inds, v_inds = [], []
                self.train_folds, self.valid_folds = [], []

                # weights = torch.DoubleTensor(self.ds.weights.values.flatten())
                # _sampler = sampler.WeightedRandomSampler(weights, len(weights))

                # train_indices, test_indices = train_test_split(list(range(len(self.ds))), test_size=test_size, shuffle=True)
                # train_ds = Subset(self.ds, train_indices)
                # self.test = DataLoader(Subset(self.ds, test_indices), **kwargs)

                splitter = StratifiedKFold(n_splits=self.kfolds, shuffle=True, )
                # splitter = KFold(n_splits=self.kfolds, shuffle=True, )
                for fold, (train_ind, val_ind) in enumerate(splitter.split(self.ds.X,
                                                                           self.ds.Y_single)):
                    weights = torch.DoubleTensor(self.ds.weights.values.flatten())[train_ind]
                    _sampler = sampler.WeightedRandomSampler(weights, len(weights), replacement=True)
                    print(weights)
                    self.train_folds.append(DataLoader(Subset(self.ds, train_ind), sampler=_sampler, drop_last=True, **kwargs))
                    self.valid_folds.append(DataLoader(Subset(self.ds, val_ind), drop_last=True, **kwargs))
                    t_inds.append(self.train_folds[fold].dataset.indices)
                    v_inds.append(self.valid_folds[fold].dataset.indices)

                t_inds = pd.DataFrame(t_inds)
                v_inds = pd.DataFrame(v_inds)
                t_inds.to_csv('t_inds.csv', index=False)
                v_inds.to_csv('v_inds.csv', index=False)
        else:
            self.ds = DataLoader(self.ds, shuffle=False, **kwargs)

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return ('<Custom pyTorch DataLoader:\n\t'
                f'Total Size: {len(self.ds)}\n\t\t'
                f'-Training set Size: {len(self.train)}\n\t\t'
                f'-Validation set Size: {len(self.valid)}\n\t\t'
                f'-Test set Size: {len(self.test)}'
                '>')

    @staticmethod
    def get_fold_indices(folds):
        fold_indices = []
        for f in folds:
            fold_indices.extend(f.indices)
        return fold_indices




# ds = AudioDS(non_binary=True, percent_weights=True)
# ds = AudioDS()
# ds = AudioDS(mode='infer')
#
# i = 0
# for x, y in ds:
    # print(ds.Y_single[i], y)
    # i+=1
# # print(np.unique(DS.Y_single, return_counts=True))
# print(len(ds))
# for x in ds:
#     print(x)

# ds = Loader(cross_valid=True, kfolds=6)
# # print(len(ds.train_folds))
# # # # print(dir(ds))
# ys = []
# for fold in ds.train_folds:
#     # print(len(fold))
#     for x, y in fold:
#         ys.extend(torch.argmax(y, dim=-1).flatten())
#
#     # print(ys)
#     ys = pd.DataFrame(np.array(ys))
#     ys.to_csv('ys.csv', index=False)


# ds = Loader()
# for x, y in ds.train:
#     print(x, y)