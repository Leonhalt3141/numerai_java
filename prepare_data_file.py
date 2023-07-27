import sys
import gc

import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file


def single_quotation(df: pd.DataFrame):
    new_cols = [c.replace("'", "") for c in df.columns]
    df.columns = new_cols


def convert_datatype(df: pd.DataFrame):
    for c in df.columns:
        if "feature" in c or "target" in c:
            df[c] = df[c].astype(np.float32)


def basic_fix(df: pd.DataFrame):
    single_quotation(df)
    convert_datatype(df)


def get_feature_names(df):
    return [c for c in df.columns if 'feature' in c]


def convert_train(train_parquet: str):
    df = pd.read_parquet(train_parquet)
    basic_fix(df)

    x_names = get_feature_names(df)
    df = df[x_names + ["target"]]
    gc.collect()

    # df.dropna(inplace=True)
    df.fillna(0, inplace=True)
    dump_svmlight_file(df[x_names], df["target"], train_parquet.replace("parquet", "libsvm"))
    del df
    gc.collect()


def convert_live(live_parquet: str):
    df = pd.read_parquet(live_parquet)
    basic_fix(df)

    x_names = get_feature_names(df)
    df = df[x_names + ["target"]]
    df.fillna(0, inplace=True)

    dump_svmlight_file(df[x_names], df['target'], live_parquet.replace("parquet", "livsvm"))
    del df
    gc.collect()


if __name__ == "__main__":
    args = sys.argv

    train_parquet = args[1]
    live_parquet = args[2]

    print(f"Converting train parquet: {train_parquet}")
    convert_train(train_parquet)

    print(f"Converting live parquet: {live_parquet}")
    convert_live(live_parquet)
