import pandas as pd
from convert import concat_dataframe
from autogluon.tabular import TabularPredictor
import numpy as np
import heapq
from typing import List


def get_max(index) -> pd.DataFrame:
    df = pd.read_excel('data/train_1.xlsx', f'材料{index}')
    df.columns = ['Temperature', 'Frequency', 'Core_loss', 'Wave_type'] + [f'B_field{i}' for i in range(1024)]
    df = df.drop(columns=['Core_loss', 'Wave_type', 'Temperature', 'Frequency'], axis=1, inplace=False)
    # print(df.info())
    a = df.max(axis=1)
    return a


def concat_max():
    df_list = [get_max(i) for i in range(1, 5)]
    result = pd.concat(df_list, ignore_index=True)
    return result


def run_concat():
    b_max = concat_max()
    data = concat_dataframe()
    data['b_max'] = b_max
    return data


def obj_fun() -> List:
    def norm(arr: np.ndarray) -> np.ndarray:
        _range = np.max(arr) - np.min(arr)
        return (arr - np.min(arr)) / _range

    data = run_concat()
    w1, w2 = 2, -1
    predictor = TabularPredictor.load('agModels_Regression')
    pred_core_loss = predictor.predict(data.drop(columns=['b_max', 'Core_loss'], axis=1, inplace=False))
    # print(pred_core_loss.describe())
    pred_core_loss = np.array(pred_core_loss)
    freq = np.array(data['Frequency'])
    b_max = np.array(data['b_max'])
    res = w1 * norm(freq) * norm(b_max) + w2 * norm(pred_core_loss)
    res = res.tolist()
    tmp = zip(range(len(res)), res)
    large10 = heapq.nlargest(10, tmp, key=lambda x: x[1])
    ret = []
    for item in large10:
        index, value = item
        tmp= data.iloc[index, [0, 1, 2, 4, -1]].values.tolist()
        tmp.append(value)
        ret.append(tmp)
    return ret


def write_to_csv():
    res = obj_fun()
    df = pd.DataFrame(res, columns=['Material', 'Temperature', 'Frequency', 'Wave_type', 'Max_b', 'Scores'])
    df['Material'] = df['Material'].apply(lambda x: f'材料{int(x)}')
    df['Temperature'] = df['Temperature'].astype(np.int64)
    df['Wave_type'] = df['Wave_type'].astype(np.int64)
    df.loc[df['Wave_type'] == 0, 'Wave_type'] = '正弦波'
    df.loc[df['Wave_type'] == 1, 'Wave_type'] = '三角波'
    df.loc[df['Wave_type'] == 2, 'Wave_type'] = '梯形波'
    df.to_csv('src/final.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    write_to_csv()