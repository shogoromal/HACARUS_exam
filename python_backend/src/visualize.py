import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def plot_histograms(dataframes:List[pd.DataFrame], names:List[int], graph_name):
    # 全てのDataFrameに共通するカラム名を取得
    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns.intersection_update(df.columns)
    print(common_columns)
    # 共通のカラムごとにヒストグラムをプロット
    for column in common_columns:
        plt.figure(figsize=(10, 6))
        for df, name in zip(dataframes, names):
            if column in df.columns:
                # データをプロット
                df[column].hist(alpha=0.5, label=name, bins=15)
        
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('outputs/'+graph_name+'.jpg')
        