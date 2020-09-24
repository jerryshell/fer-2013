import itertools

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


class DataLoader:
    def __init__(self):
        # 加载数据
        data = pd.read_csv(config.train_data_filepath)
        print('data.shape', data.shape)
        print(data.describe())

        # pandas 转 numpy
        data = data.to_numpy()
        data_count = len(data)

        # 打乱数据集
        np.random.shuffle(data)

        # 切分训练集、验证集、测试集 8:1:1
        self.train_data, self.validation_data, self.test_data = np.vsplit(
            data,
            [int(data_count * 0.8), int(data_count * 0.9)]
        )
        print('train_data.shape', self.train_data.shape)
        print('validation_data.shape', self.validation_data.shape)
        print('test_data.shape', self.test_data.shape)

        # 记录训练集、验证集、测试集的大小
        self.train_data_count = len(self.train_data)
        self.validation_data_count = len(self.validation_data)
        self.test_data_count = len(self.test_data)
        print('train_data_count', self.train_data_count)
        print('validation_data_count', self.validation_data_count)
        print('test_data_count', self.test_data_count)

        # # train_data_x
        # self.train_data_x = self.train_data[:, 1]
        # self.train_data_x = [x.split(' ') for x in self.train_data_x]
        # self.train_data_x = [list(map(int, x)) for x in self.train_data_x]
        # self.train_data_x = np.reshape(self.train_data_x, (len(self.train_data_x), 48, 48, 1))
        #
        # # train_data_y
        # self.train_data_y = self.train_data[:, 0]

    def train_data_generator(self):
        return itertools.cycle(self.train_data)

    def validation_data_generator(self):
        return itertools.cycle(self.validation_data)

    def test_data_generator(self):
        return itertools.cycle(self.test_data)


if __name__ == '__main__':
    data_loader = DataLoader()

    train_data_generator = data_loader.train_data_generator()
    validation_data_generator = data_loader.validation_data_generator()
    test_data_generator = data_loader.test_data_generator()

    # print(next(train_data_generator))
    # print(next(train_data_generator))
    # print('---')
    # print(next(validation_data_generator))
    # print(next(validation_data_generator))
    # print('---')
    # print(next(test_data_generator))
    # print(next(test_data_generator))

    # train_data_x = data_loader.train_data[:, 1]
    # print(train_data_x[0])
    # print(train_data_x.shape)
    # train_data_x = [map(int, x.split(' ')) for x in train_data_x]
    # print(train_data_x[0])

    # print(data_loader.train_data_x[0])
    # plt.imshow(data_loader.train_data_x[0])
    # plt.show()
    # print(len(data_loader.train_data_x))
    # print(data_loader.train_data_y[0])
    # print(len(data_loader.train_data_y))
