import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sko.GA import GA
import torch
from sko.operators import ranking, crossover, mutation, selection
from sko.operators.selection import selection_tournament
from matplotlib import pyplot as plt


def read_data(is_shuffle=True):
    data, labels = [], []
    df = pd.read_excel("data/train_1.xlsx", sheet_name="材料1", header=0)
    tmp = df.values.tolist()
    if is_shuffle:
        random.shuffle(tmp)
    for row in tmp:
        if row[3] == "正弦波":
            b_max = max(row[4:])
            new_row = row[:2]
            new_row.append(b_max)
            data.append(new_row)
            labels.append(row[2])
    return data, labels


def split_data(data, labels):
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    return x_train, x_val, y_train, y_val


def get_array(data):
    temp = [row[0] for row in data]
    freq = [row[1] for row in data]
    max_b = [row[2] for row in data]
    return np.array(temp, dtype=float), np.array(freq, dtype=float), np.array(max_b, dtype=float)


def get_label_array(labels):
    return np.array(labels, dtype=float)


def steinmetz_eq(data, params):
    temp, freq, max_b = get_array(data)
    a, b, alpha, beta = params
    return (a * (temp + 273.15) ** b) * (freq ** alpha) * (max_b ** beta)


def obj_func(params):  # Attention: the function must be used after the x_train and y_train declared.
    residuals = np.mean(np.square(steinmetz_eq(x_train, params) - y_train))
    return residuals


def st_eq(test_data, test_params):
    _, freq, max_b = get_array(test_data)
    k, alpha, beta = test_params
    return k * freq ** alpha * max_b ** beta


def obj_func_2(params):
    residuals = np.mean(np.square(st_eq(x_train, params) - y_train))
    return residuals

def compare(x_val, y_val, best_params_new, best_params_old):
    y_pred = steinmetz_eq(x_val, best_params_new)
    y_pred_old = st_eq(x_val, best_params_old)
    new_residuals = np.mean(np.square(y_val - y_pred))
    old_residuals = np.mean(np.square(y_val - y_pred_old))
    print(f'\nnew_residuals: {new_residuals}, old_residuals: {old_residuals}\n')


def plot_spots(y_pred, y_pred_old, y_actual):
    plt.figure(figsize=(10, 10), dpi=500)
    plt.scatter(range(len(y_pred)), y_pred, label="prediction of new method", color="blue")
    plt.scatter(range(len(y_actual)), y_actual, label="actual core loss", color="red")
    plt.scatter(range(len(y_pred_old)), y_pred_old, label="prediction of old method", color="green")
    plt.xlabel("Sample index")
    plt.ylabel("Core loss")
    plt.legend()
    plt.grid(True)
    plt.title("Prediction Loss vs Actual Loss")
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, labels = read_data()
    x_train, x_val, y_train, y_val = split_data(data, labels)
    ga = GA(func=obj_func, n_dim=4, size_pop=200, max_iter=5000, prob_mut=0.01,
            lb=[-10, -3, 1, 2], ub=[20000000, 2, 3, 3])
    ga.register(operator_name='ranking', operator=ranking.ranking). \
        register(operator_name='crossover', operator=crossover.crossover_2point). \
        register(operator_name='mutation', operator=mutation.mutation). \
        register(operator_name='selection', operator=selection_tournament, tourn_size=3)
    ga.to(device)
    best_params, residuals = ga.run()

    ga_2 = GA(func=obj_func_2, n_dim=3, size_pop=200, max_iter=5000, prob_mut=0.01,
            lb=[0, 1, 2], ub=[20000000, 3, 3])
    ga_2.register(operator_name='ranking', operator=ranking.ranking). \
        register(operator_name='crossover', operator=crossover.crossover_2point). \
        register(operator_name='mutation', operator=mutation.mutation). \
        register(operator_name='selection', operator=selection_tournament, tourn_size=3)
    ga_2.to(device)
    best_params_old, residuals_old = ga_2.run()

    # print('best_params:', best_params, 'residuals:', residuals)
    print(f'best_params:{best_params}, residuals:{residuals}\n')
    print(f'best_params_old:{best_params_old}, residuals_old:{residuals_old}\n')

    compare(x_val, y_val, best_params, best_params_old)
    plot_spots(steinmetz_eq(x_val, best_params), st_eq(x_val, best_params_old), y_val)
