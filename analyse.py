import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt


def cal_probability(labels):
    n = len(labels)
    hash_list = Counter(labels)
    print(hash_list)
    sin_wave = hash_list.get("正弦波") / n
    tri_wave = hash_list.get("三角波") / n
    tra_wave = 1 - sin_wave - tri_wave
    return np.array([sin_wave, tri_wave, tra_wave])


def plot_material_to_loss(loss):
    x = [f"material {i}" for i in range(1, len(loss) + 1)]
    plt.figure(figsize=(8, 10), dpi=100)
    plt.bar(x , loss, color="blue", width=0.2)
    plt.title("Average Magnetic Flux Density in Different Materials")
    plt.xlabel("Materials")
    plt.ylabel("Average magnetic flux density / T")
    plt.show()


def plot_wave_to_loss(loss):
    x = [f'material {i}' for i in range(1, len(loss) + 1)]
    x1 = np.arange(len(x))
    x2 = [x + 0.2 for x in x1]
    x3 = [x + 0.2 for x in x2]
    plt.figure(figsize=(12, 12), dpi=100)
    plt.bar(x1, [i[0] for i in loss], color='blue', width=0.2, label='sine', edgecolor='white')
    plt.bar(x2, [i[1] for i in loss], color='green', width=0.2, label='triangle', edgecolor='white')
    plt.bar(x3, [i[2] for i in loss], color='red', width=0.2, label='trapezoidal', edgecolor='white')
    plt.xticks([r + 0.2 for r in range(len(x))], x)
    plt.xlabel("Different Waves on materials")
    plt.ylabel("Average magnetic flux density / T")
    plt.title("Average Magnetic Flux Density Divided by Waves")
    plt.legend()
    plt.show()


def plot_temperature_to_loss(loss):
    x = [f'material {i}' for i in range(1, len(loss) + 1)]
    bar_width = 0.2
    x1 = np.arange(len(x))
    x2 = [x + bar_width for x in x1]
    x3 = [x + bar_width for x in x2]
    x4 = [x + bar_width for x in x3]
    plt.figure(figsize=(12, 12), dpi=100)
    plt.bar(x1, [i[0] for i in loss], color='blue', width=bar_width, label='25', edgecolor='white')
    plt.bar(x2, [i[1] for i in loss], color='green', width=bar_width, label='50', edgecolor='white')
    plt.bar(x3, [i[2] for i in loss], color='red', width=bar_width, label='70', edgecolor='white')
    plt.bar(x4, [i[3] for i in loss], color='yellow', width=bar_width, label='90', edgecolor='white')
    plt.xticks([r + bar_width for r in range(len(x))], x)
    plt.xlabel("Different Temperature on materials / ℃")
    plt.ylabel("Average magnetic flux density / T")
    plt.title("Average Magnetic Flux Density Divided by Temperature")
    plt.legend()
    plt.show()


class Analyse:
    def __init__(self):
        self.data_dir = "data/train_1.xlsx"

    def read_sheet(self, index):
        data = []
        sheet_name = f"材料{index}"
        df = pd.read_excel(self.data_dir, sheet_name=sheet_name, header=0)
        tmp = df.values.tolist()
        for row in tmp:
            row = row[:4]
            row.pop(1)
            data.append(row)
        return data

    def run_material(self):
        losses = []
        for i in tqdm(range(1, 5)):
            losses.append(np.mean(self.get_core_loss(i)))
        print(losses)
        return losses

    def get_core_loss(self, index):
        data = self.read_sheet(index)
        core_loss = np.array([i[1] for i in data])
        return core_loss

    def get_core_loss_by_waves(self, index):
        data = self.read_sheet(index)
        sine_loss = np.array([i[1] for i in data if i[-1] == "正弦波"])
        triangle_loss = np.array([i[1] for i in data if i[-1] == "三角波"])
        trapezoidal_loss = np.array([i[1] for i in data if i[-1] == "梯形波"])
        return sine_loss, triangle_loss, trapezoidal_loss

    def run_wave(self):
        losses = []
        for i in tqdm(range(1, 5)):
            loss = []
            a, b, c = self.get_core_loss_by_waves(i)
            loss.append(np.mean(a))
            loss.append(np.mean(b))
            loss.append(np.mean(c))
            losses.append(loss)
        return losses

    def get_core_loss_by_temperature(self, index):
        data = self.read_sheet(index)
        temp_25 = np.array([i[1] for i in data if i[0] == 25])
        temp_50 = np.array([i[1] for i in data if i[0] == 50])
        temp_70 = np.array([i[1] for i in data if i[0] == 70])
        temp_90 = np.array([i[1] for i in data if i[0] == 90])
        return temp_25, temp_50, temp_70, temp_90

    def run_temperature(self):
        losses = []
        for i in tqdm(range(1, 5)):
            loss = []
            a, b, c, d = self.get_core_loss_by_temperature(i)
            loss.append(np.mean(a))
            loss.append(np.mean(b))
            loss.append(np.mean(c))
            loss.append(np.mean(d))
            losses.append(loss)
        return losses


def run():
    loss_materials = Analyse().run_material()
    plot_material_to_loss(loss_materials)

    loss_waves = Analyse().run_wave()
    plot_wave_to_loss(loss_waves)

    loss_temperature = Analyse().run_temperature()
    plot_temperature_to_loss(loss_temperature)


if __name__ == "__main__":
    # run()
    pass
