import numpy as np

from data_loader import DataLoader


class DataHelper:
    def __init__(self):
        # 加载原始数据
        data_loader = DataLoader()

        # 原始训练数据生成器
        self.raw_train_data_generator = data_loader.train_data_generator()
        # 原始验证数据生成器
        self.raw_validation_data_generator = data_loader.validation_data_generator()
        # 原始测试数据生成器
        self.raw_test_data_generator = data_loader.test_data_generator()

        # 训练数据大小
        self.train_data_count = data_loader.train_data_count
        # 验证数据大小
        self.validation_data_count = data_loader.validation_data_count
        # 测试数据大小
        self.test_data_count = data_loader.test_data_count

    # data_generator 中读取 batch_size 个数据
    def get_batch_pixel_list_and_label(self, data_generator: iter, batch_size: int):
        # 初始化返回结果
        batch_label = np.zeros(shape=(batch_size,))
        batch_pixel_list = np.zeros(shape=(batch_size, 48, 48))
        # 根据 batch_size 填充返回结果
        for batch_index in range(batch_size):
            # 从 data_generator 中读取下一个数据
            label, pixel_raw_data = next(data_generator)
            # pixel_raw_data2pixel_list
            pixel_list = self.pixel_raw_data2pixel_list(pixel_raw_data)
            # pixel_list_reshape
            pixel_list_reshape = self.pixel_list_reshape(pixel_list)
            # 填充返回结果
            batch_label[batch_index] = label
            batch_pixel_list[batch_index] = pixel_list_reshape
        return batch_pixel_list, batch_label

    # 训练数据生成器
    def train_data_generator(self, batch_size: int):
        while True:
            yield self.get_batch_pixel_list_and_label(self.raw_train_data_generator, batch_size)

    # 验证数据生成器
    def validation_data_generator(self, batch_size: int):
        while True:
            yield self.get_batch_pixel_list_and_label(self.raw_validation_data_generator, batch_size)

    # 测试数据生成器
    def test_data_generator(self, batch_size: int):
        while True:
            yield self.get_batch_pixel_list_and_label(self.raw_test_data_generator, batch_size)

    def pixel_raw_data2pixel_list(self, pixel_raw_data: str):
        return [int(item) for item in pixel_raw_data.split(' ')]

    def pixel_list_reshape(self, pixel_list: list):
        return np.reshape(pixel_list, (48, 48))


if __name__ == '__main__':
    data_helper = DataHelper()
    data_loader = DataLoader()

    for index, data in enumerate(data_loader.train_data):
        print('---')
        print('index', index)

        label = data[0]
        pixel_raw_data = data[1]
        print('pixel_raw_data', pixel_raw_data)

        pixel_list = data_helper.pixel_raw_data2pixel_list(pixel_raw_data)
        print('pixel_list', pixel_list)

        pixel_list_reshape = data_helper.pixel_list_reshape(pixel_list)
        print('pixel_list_reshape', pixel_list_reshape)

        if index == 5:
            break
