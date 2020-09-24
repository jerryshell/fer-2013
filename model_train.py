from datetime import datetime

import tensorflow.keras as keras

from data_helper import DataHelper


def train(
        data_helper: DataHelper,
        model: keras.Model,
        save_filename: str,
        batch_size: int = 64,
        epochs: int = 10
):
    # 模型信息
    model.summary()

    # model file name
    val_acc_best_model_filename = save_filename + '.val.acc.best.h5'
    final_model_filename = save_filename + '.final.h5'

    # 训练数据生成器
    train_data_generator = data_helper.train_data_generator(batch_size)
    # 验证数据生成器
    validation_data_generator = data_helper.validation_data_generator(batch_size)
    # 测试数据生成器
    test_data_generator = data_helper.test_data_generator(batch_size)

    # 训练
    model.fit(
        x=train_data_generator,
        steps_per_epoch=data_helper.train_data_count // batch_size,
        validation_data=validation_data_generator,
        validation_steps=data_helper.validation_data_count // batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=[
            # 配置 tensorboard，将训练过程可视化，方便调参，tensorboard --logdir logs/fit
            keras.callbacks.TensorBoard(
                log_dir='logs/fit/' + datetime.now().strftime('%Y%m%d-%H%M%S'),
                histogram_freq=1
            ),
            # 定时保存模型
            keras.callbacks.ModelCheckpoint(
                filepath=val_acc_best_model_filename,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                save_freq='epoch'
            )
        ],
    )

    # 最终模型测试
    model.evaluate(
        x=test_data_generator,
        steps=data_helper.test_data_count // batch_size,
    )

    # 保存最终模型
    model.save(filepath=final_model_filename)

    # final model evaluate
    print('final model evaluate')
    final_model = keras.models.load_model(filepath=final_model_filename)
    final_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )
    final_model.evaluate(
        x=test_data_generator,
        steps=data_helper.test_data_count // batch_size,
    )

    # val_acc best 模型测试
    print('val_acc best evaluate')
    val_acc_best_model = keras.models.load_model(filepath=val_acc_best_model_filename)
    val_acc_best_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['acc'],
    )
    val_acc_best_model.evaluate(
        x=test_data_generator,
        steps=data_helper.test_data_count // batch_size,
    )


if __name__ == '__main__':
    import sys
    import model_creator

    print(sys.argv)
    if len(sys.argv) != 3:
        print('python3 model_train.py <batch_size> <epochs>')
        print('example: python3 model_train.py 64 10')
        exit(0)

    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    print('batch_size %s epochs %s' % (batch_size, epochs))

    # 加载数据
    print('data loading...')
    data_helper = DataHelper()

    # 重新训练一个新模型
    model = model_creator.create_model()
    train(
        data_helper=data_helper,
        model=model,
        save_filename='fer.' + datetime.now().strftime('%Y%m%d-%H%M%S'),
        batch_size=batch_size,
        epochs=epochs
    )
