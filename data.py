from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load():
    # data augmentation a, b <=> b, a
    data_dir = Path("comboFM_core_data/data/data/")
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
            pd.read_csv(path / csv)
            for csv in tqdm(csvs)
        ]
        return pd.concat(dfs, axis=1)

    dff = csvs_load(data_dir, forward)
    dfb = csvs_load(data_dir, backward)
    dfb.columns = list(dff.columns)
    df = pd.concat((dff, dfb), axis=0)
    return df


def split(df, mode, inner_fold, outer_fold):
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

    rest_idx = pd.read_csv(
        f'comboFM_core_data/cross-validation_folds'
        f'/{mode}/train_idx_outer_fold-{outer_fold}.txt',
        header=None).values
    test_idx = pd.read_csv(
        f'comboFM_core_data/cross-validation_folds'
        f'/{mode}/test_idx_outer_fold-{outer_fold}.txt',
        header=None).values

    df_test = df.iloc[test_idx.flatten()]
    # df_rest = df.iloc[rest_idx.flatten()]

    train_idx = pd.read_csv(
        f'comboFM_core_data/cross-validation_folds'
        f'/{mode}/train_idx_outer_fold-{outer_fold}'
        f'_inner_fold-{inner_fold}.txt',
        header=None).values
    dev_idx = pd.read_csv(
        f'comboFM_core_data/cross-validation_folds'
        f'/{mode}/test_idx_outer_fold-{outer_fold}'
        f'_inner_fold-{inner_fold}.txt',
        header=None).values
    df_train = df.iloc[train_idx.flatten()]
    df_dev = df.iloc[dev_idx.flatten()]

    return df_train, df_dev, df_test

class ScaleAbsOne:
    def __init__(self):
        self.axis = None
        self.max = 0

    def fit(self, x, axis=0):
        self.axis = axis
        self.max = x.max(axis=axis, keepdims=True)

    def transform(self, x):
        return x / self.max

    def fit_transform(self, x, axis=0):
        self.fit(x, axis)
        return self.transform(x)


if __name__ == "__main__":
    df = load()
    train, dev, test = split(df, "new_dose-response_matrices", 1, 1)
    train_target = train.pop("PercentageGrowth")
    dev_target = dev.pop("PercentageGrowth")
    test_target = test.pop("PercentageGrowth")