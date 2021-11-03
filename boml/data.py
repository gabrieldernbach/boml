from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


def load(debug=False):
    # data augmentation a, b <=> b, a
    data_dir = Path("comboFM_data/data/")
    forward = (
        "drug1_concentration__one-hot_encoding.csv",
        "drug2_concentration__one-hot_encoding.csv",
        "drug1__one-hot_encoding.csv",
        "drug2__one-hot_encoding.csv",
        "cell_lines__one-hot_encoding.csv",
        "drug1_drug2_concentration__values.csv",
        "drug1__estate_fingerprints.csv",
        "drug2__estate_fingerprints.csv",
        "cell_lines__gene_expression.csv",
        "responses.csv"
    )
    backward = (
        "drug2_concentration__one-hot_encoding.csv",
        "drug1_concentration__one-hot_encoding.csv",
        "drug2__one-hot_encoding.csv",
        "drug1__one-hot_encoding.csv",
        "cell_lines__one-hot_encoding.csv",
        "drug2_drug1_concentration__values.csv",
        "drug2__estate_fingerprints.csv",
        "drug1__estate_fingerprints.csv",
        "cell_lines__gene_expression.csv",
        "responses.csv"
    )

    def csvs_load(path, csvs):
        dfs = [
            pd.read_csv(path / csv, nrows=5000 if debug else None)
            for csv in tqdm(csvs)
        ]
        return pd.concat(dfs, axis=1)

    dff = csvs_load(data_dir, forward)
    dfb = csvs_load(data_dir, backward)
    dfb.columns = list(dff.columns)
    df = pd.concat((dff, dfb), axis=0)
    return df


def split(df, mode, inner_fold, outer_fold, debug=False):
    modes = (
        "new_dose-response_matrix_entries",
        "new_dose-response_matrices",
        "new_drug_combinations"
    )
    if mode not in modes:
        raise ValueError(f"mode must be either of {modes}")
    if not (type(outer_fold) == int) or not (1 <= outer_fold <= 10):
        raise ValueError(f"outer_fold must be int in [1, 10])")
    if not (type(inner_fold) == int) or not (1 <= inner_fold <= 5):
        raise ValueError(f"inner_fold must be int in [1, 10])")

    # rest_idx = pd.read_csv(
    #    f'comboFM_data/cross-validation_folds'
    #    f'/{mode}/train_idx_outer_fold-{outer_fold}.txt',
    #    header=None).values
    test_idx = pd.read_csv(
        f'comboFM_data/cross-validation_folds'
        f'/{mode}/test_idx_outer_fold-{outer_fold}.txt',
        header=None).values.flatten()

    train_idx = pd.read_csv(
        f'comboFM_data/cross-validation_folds'
        f'/{mode}/train_idx_outer_fold-{outer_fold}'
        f'_inner_fold-{inner_fold}.txt',
        header=None).values.flatten()
    dev_idx = pd.read_csv(
        f'comboFM_data/cross-validation_folds'
        f'/{mode}/test_idx_outer_fold-{outer_fold}'
        f'_inner_fold-{inner_fold}.txt',
        header=None).values.flatten()
    if debug:
        test_idx = test_idx[test_idx < df.shape[0]]
        train_idx = train_idx[train_idx < df.shape[0]]
        dev_idx = dev_idx[dev_idx < df.shape[0]]
    df_test = df.iloc[test_idx]
    # df_rest = df.iloc[rest_idx.flatten()]
    df_train = df.iloc[train_idx]
    df_dev = df.iloc[dev_idx]

    return df_train, df_dev, df_test


class ScaleAbsOne:
    def __init__(self):
        self.axis = None
        self.max = 0

    def fit(self, x, axis=0):
        self.axis = axis
        self.max = abs(x).max(axis=axis, keepdims=True)

    def transform(self, x):
        return x / self.max

    def fit_transform(self, x, axis=0):
        self.fit(x, axis)
        return self.transform(x)


def load_dataloaders(label="PercentageGrowth", debug=False, batch_size=10):
    df = load(debug=debug)
    train_set, dev_set, test_set = split(df, "new_dose-response_matrices", inner_fold=1, outer_fold=1, debug=debug)
    train_target = train_set.pop(label)
    dev_target = dev_set.pop(label)
    test_target = test_set.pop(label)
    scaler = ScaleAbsOne()
    std_scaler = StandardScaler()
    train_target = std_scaler.fit_transform(train_target.values[:, None])[:, 0]
    dev_target = std_scaler.transform(dev_target.values[:, None])[:, 0]
    test_target = std_scaler.transform(test_target.values[:, None])[:, 0]
    train_set = scaler.fit_transform(train_set.values)
    dev_set = scaler.transform(dev_set.values)
    test_set = scaler.transform(test_set.values)

    dataloaders = {}
    datasets = {'train': [train_set, train_target], 'dev': [dev_set, dev_target],
                'test': [test_set, test_target]}
    for typ in datasets:
        t = datasets[typ]
        if debug:
            import numpy as np
            t = [np.nan_to_num(t[0]), np.nan_to_num(t[1])]
        dataset = TensorDataset(torch.tensor(t[0]).float(), torch.tensor(t[1]).float())
        dataloaders[typ] = DataLoader(dataset, batch_size=batch_size, num_workers=6)

    return [dataloaders[typ] for typ in ['train', 'dev', 'test']]


if __name__ == "__main__":
    df = load()
    train, dev, test = split(df, "new_dose-response_matrices", 1, 1)
    train_target = train.pop("PercentageGrowth")
    dev_target = dev.pop("PercentageGrowth")
    test_target = test.pop("PercentageGrowth")
