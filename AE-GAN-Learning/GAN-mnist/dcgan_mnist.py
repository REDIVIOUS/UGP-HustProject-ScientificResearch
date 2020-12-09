import tensorflow as tf
from tensorflow.keras import datasets, Sequential, layers, losses
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)

def draw():
    plt.figure()
    plt.plot(g_losses, 'b', label='generator')
    plt.plot(d_losses, 'r', label='discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig('Anime_dcgan_ACC.png')

d_losses, g_losses = [], []

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# 观察mnist数据类型
print(x_train, y_train, x_test, y_test)

# 4 * 4 * 7 => 28 * 28 * 1
# (60000, 28, 28) - (10000, 28, 28)
generator = Sequential([
    layers.Dense(4 * 4 * 7, activation=tf.nn.leaky_relu),
    layers.Reshape(target_shape=(4, 4, 7)),
    layers.Conv2DTranspose(14, 5, 2, activation=tf.nn.leaky_relu),
    layers.BatchNormalization(),
    layers.Conv2DTranspose(5, 3, 1, activation=tf.nn.leaky_relu),
    layers.BatchNormalization(),
    layers.Conv2DTranspose(1, 4, 2, activation=tf.nn.tanh),
    layers.Reshape(target_shape=(28, 28)),
])

discriminator = Sequential([
    layers.Reshape((28, 28, 1)),
    layers.Conv2D(3, 4, 2, activation=tf.nn.leaky_relu),
    layers.BatchNormalization(),
    layers.Conv2D(12, 3, 1, activation=tf.nn.leaky_relu),
    layers.BatchNormalization(),
    layers.Conv2D(28, 5, 2, activation=tf.nn.leaky_relu),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(1)
])
# 5s 89us/sample - loss: 0.0264 - accuracy: 0.9949 - val_loss: 0.1412 - val_accuracy: 0.9863

# 超参数
dim_h = 100
epochs = 100
batch_size = 128
learning_rate = 2e-3


def preprocess(pre_x, pre_y):
    pre_x = tf.cast(pre_x, dtype=tf.float32) / 255.
    pre_y = tf.cast(pre_y, dtype=tf.int32)
    return pre_x, pre_y


db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .map(preprocess).shuffle(batch_size * 5).batch(batch_size, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    .map(preprocess).shuffle(batch_size * 5).batch(batch_size, drop_remainder=True)

generator.build((None, dim_h))
generator.summary()

discriminator.build((None, 28, 28, 1))
discriminator.summary()

# 是不是对应的
print(generator(tf.random.normal((1, dim_h))))
print(discriminator(tf.random.normal((1, 28, 28, 1))))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
cross_entropy = losses.BinaryCrossentropy(from_logits=True)

def main():
    for epoch in range(epochs):
        for step, (true_x, y) in enumerate(db_train):
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                # 随机一个标准的种子
                random_seek = tf.random.normal((batch_size, dim_h))
                # 生成一批假图片
                false_x = generator(random_seek)
                # 通过判断器鉴别假图片
                false_y = discriminator(false_x)
                true_y = discriminator(true_x)
                false_loss = cross_entropy(tf.zeros_like(false_y), false_y)
                true_loss = cross_entropy(tf.ones_like(true_y), true_y)
                d_loss = false_loss + true_loss
                g_loss = cross_entropy(tf.ones_like(false_y), false_y)
            d_grad = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))
            g_grad = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))

        # 记录loss
        d_losses.append(float(d_loss))
        g_losses.append(float(g_loss))

        if epoch % 5 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))
            # 打印一张图片
            z = tf.random.normal([100, dim_h])
            fake_image = generator(z, training=False)
            if not os.path.exists('mnist-images'):
                os.mkdir('mnist-images')
            img_path = os.path.join('mnist-images', 'gan-one%d.png' % epoch)
            fake_image = tf.expand_dims(fake_image, axis=3)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

if __name__ == '__main__':
    main()
    draw()