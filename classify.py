import pandas as pd
from typing import List, Tuple
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
from collections import Counter


def get_key(value) -> str:
    name_dict = {
        "正弦波": 0,
        "三角波": 1,
        "梯形波": 2,
    }
    return [k for k, v in name_dict.items() if v == value][0]


def read(data_loc: str, sheet_name: str):
    df = pd.read_excel(data_loc, sheet_name=sheet_name)
    print(df)


class Process:
    def __init__(self, data_loc: str, sheet_name: str, is_shuffle: bool = True):
        self.data_loc = data_loc
        self.sheet_name = sheet_name
        self.is_shuffle = is_shuffle
        self.labels = []
        self.data = []

    def read_data(self):
        df = pd.read_excel(self.data_loc, sheet_name=self.sheet_name, header=0)
        data = df.values.tolist()
        if self.is_shuffle:
            random.shuffle(data)
        for row in data:
            self.labels.append(row[3])
            self.data.append(row[4:])
        # print(self.data[0], self.labels[0])

    def process(self) -> (List, List):
        self.read_data()
        labels = []
        for label in self.labels:
            if label == "正弦波":
                labels.append(0)
            elif label == "三角波":
                labels.append(1)
            else:
                labels.append(2)
        return self.data, labels


def plot_data(points: List[float], label: int):
    x = list(range(len(points)))
    plt.scatter(x, points)
    # plt.title(get_key(label))
    if label == 0:
        plt.title("Sine")
    elif label == 1:
        plt.title("Triangle")
    else:
        plt.title("Trapezoidal")
    plt.show()


def get_test_data(test_loc="data/test_1.xlsx", split_range=1):
    assert 0 < split_range < 5, "split_range should be in [1, 4]."
    test_data = []
    df = pd.read_excel(test_loc, header=0)
    data = df.values.tolist()
    for row in data:
        test_data.append(row[4:])
    return test_data[(split_range - 1) * 20 : split_range * 20]


def svm_train_test(sheet_index: int) -> Tuple:
    sheet_name = "材料" + str(sheet_index)
    x, y = Process("data/train_1.xlsx", sheet_name).process()
    plot_data(x[0], y[0])
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
    svm = SVC(kernel='linear', C=15, gamma='auto', probability=True)
    svm.fit(x_train, y_train)
    acc = svm.score(x_val, y_val)
    print(acc)
    x_test = get_test_data(test_loc="data/test_1.xlsx", split_range=sheet_index)
    pred = svm.predict(x_test)
    print(pred)
    return acc, pred


def run():
    final = []
    for i in range(1, 5):
        acc, pred = svm_train_test(i)
        for x in pred:
            final.append(x)
    print(Counter(final))
    return final


if __name__ == '__main__':
    res = run()