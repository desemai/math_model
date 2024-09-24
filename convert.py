import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def convert(index):
    df = pd.read_excel('data/train_1.xlsx', sheet_name=f'材料{index}')
    df.columns = ['Temperature', 'Frequency', 'Core_loss', 'Wave_type'] + [f'B_field{index}' for index in range(1024)]
    df.loc[df['Wave_type'] == '正弦波', 'Wave_type'] = 0
    df.loc[df['Wave_type'] == '三角波', 'Wave_type'] = 1
    df.loc[df['Wave_type'] == '梯形波', 'Wave_type'] = 2
    df.insert(0, 'Material', index)
    df['Wave_type'] = df['Wave_type'].astype(np.int64)
    df['Material'] = df['Material'].astype(np.int64)
    return df


def concat_dataframe() -> pd.DataFrame:
    df_list = [convert(i) for i in range(1, 5)]
    result = pd.concat(df_list, ignore_index=True)
    return result


def store_train_test_csv(df: pd.DataFrame):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df1 = df.sample(frac=0.9, random_state=42)
    df2 = df[~df.index.isin(df1.index)]
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    print(df1.shape, df2.shape)
    print(df1.head())
    print(df2.head())
    df1.to_csv('data_convert/train.csv', index=False, encoding='utf-8', errors='strict')
    df2.to_csv('data_convert/test.csv', index=False, encoding='utf-8', errors='strict')
    return df1, df2


def plot_diff(df_train, df_test):
    dist_rows = 5
    dist_cols = 5
    plt.figure(figsize=(4 * dist_rows , 4), dpi=100)
    for i, col in enumerate(df_test.columns[:dist_rows]):
        ax = plt.subplot(1, dist_cols, i + 1)
        ax = sns.kdeplot(df_train[col], color='red', fill=True)
        ax = sns.kdeplot(df_test[col], color='blue', fill=True)
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax = ax.legend(['Train', 'Test'])
    plt.suptitle("Bias between Train Set and Test Set")
    plt.show()


def plot_relate_map(df_train):
    plt.figure(figsize=(10, 8), dpi=300)
    cols = df_train.columns[:5].tolist()
    mcorr = df_train[cols].corr(method='spearman')
    mask = np.zeros_like(mcorr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='.2f')
    plt.show()


def test_excel_to_csv():
    df = pd.read_excel('data/test_2.xlsx')
    df.columns = ['Id', 'Temperature', 'Frequency', 'Material', 'Wave_type'] + [f'B_field{i}' for i in range(1024)]
    df.loc[df['Wave_type'] == '正弦波', 'Wave_type'] = 0
    df.loc[df['Wave_type'] == '三角波', 'Wave_type'] = 1
    df.loc[df['Wave_type'] == '梯形波', 'Wave_type'] = 2
    df['Material'] = df['Material'].apply(lambda x: x[-1]).astype(np.int64)
    df['Wave_type'] = df['Wave_type'].astype(np.int64)
    df.drop(['Id'], axis=1, inplace=True)
    # print(df.info())
    df.to_csv('data_convert/submit.csv', index=False, encoding='utf-8', errors='strict')

if __name__ == '__main__':
    if not os.path.exists('data_convert/train.csv') or not os.path.exists('data_convert/test.csv'):
        store_train_test_csv(concat_dataframe())

    df_train, df_test = store_train_test_csv(concat_dataframe())

    plot_diff(df_train, df_test)
    plot_relate_map(df_train)
    test_excel_to_csv()

