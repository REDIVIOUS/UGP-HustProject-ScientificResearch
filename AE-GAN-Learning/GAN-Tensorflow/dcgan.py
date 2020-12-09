import os
import glob
import numpy as np

import tensorflow as tf
from tensorflow import keras

from gan import Generator, Discriminator
from data import make_anime_dataset

from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt


# 交叉熵函数
def celoss_ones(logits):
    # 计算属于与标签为1 的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0 的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

# 计算判别器的误差函数
# 其目的是使得L(D,G)函数最大化，使得真实样本预测为真的概率接近为1，生成样本预测为真的概率接近于0
# 所有真实样本标注为1，所有生成样本标注为0，最小化对应的交叉熵损失函数来实现最大化L(D,G)函数
def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 训练器动作
    fake_image = generator(batch_z, is_training) # 练器生成图片
    # 判定器动作
    d_fake_logits = discriminator(fake_image, is_training) # 判定器具判定生成的图片
    d_real_logits = discriminator(batch_x, is_training) # 判定器判定真实图片
    # 交叉熵计算
    d_loss_real = celoss_ones(d_real_logits) # 真实图片与1 之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits) # 生成图片与0 之间的误差
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss


# 生成器的误差函数
# 最小化交叉熵误差（后一项），迫使生成图片判定为真
def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)
    return loss

# 保存最终图片
def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])

    for b in range(val_out.shape[0]):
        # 制作行图片
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)
        # 将行图片拼接成最终图片
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    im = Image.fromarray(final_image)
    im.save('Anime_dcgan_result.png')

d_losses, g_losses = [], []


def draw():
    plt.figure()
    plt.plot(g_losses, 'b', label='generator')
    plt.plot(d_losses, 'r', label='discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig('Anime_dcgan_ACC.png')
    plt.show()


def main():
    batch_size = 64
    learning_rate = 0.0002
    z_dim = 100
    is_training = True
    epochs = 300

    img_path = glob.glob(r'data/*.png')
    print('images num:', len(img_path))
    # 构建数据集对象，返回数据集dataset和图片大小
    # (64,64,64,3) (64,64,3)
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)  # (64, 64, 64, 3) (64, 64, 3)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())  # (64, 64, 64, 3) 1.0 -1.0
    dataset = dataset.repeat(100)  # 重复循环
    db_iter = iter(dataset)

    generator = Generator()  # 创建生成器
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()  # 创建判别器
    discriminator.build(input_shape=(4, 64, 64, 3))

    # 分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    # 首先从先验分布Pz中随机采样隐藏向量，从真实数据集中随机采样真实图片
    # 通过生成器和判别器计算判别网络的损失，并优化判别器网络参数theta
    # 在训练生成器时，需要借助于判别器来计算误差，但是只计算生成器的梯度信息，并更新fi
    # 判别器训练5次之后，生成器训练一次
    for epoch in range(epochs):
        # 1. 训练判别器
        for _ in range(5):
            # 采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim]) # 采样隐藏向量
            batch_x = next(db_iter)  # 采样真实图片
            # 判别器前向计算
            with tf.GradientTape() as tape:
                # 判别器损失
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            # 判别器梯度
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            # 判别器优化
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # 2. 训练生成器
        batch_z = tf.random.normal([batch_size, z_dim]) # 采样隐藏向量
        batch_x = next(db_iter)  # 采样真实图片
        # 生成器前向计算
        with tf.GradientTape() as tape:
            # 训练器损失
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        # 训练器梯度
        grads = tape.gradient(g_loss, generator.trainable_variables)
        # 优化训练器
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        d_losses.append(float(d_loss))
        g_losses.append(float(g_loss))

        # 阶段性显示loss
        if epoch % 10 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))  # 可视化
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('gan_images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()
    draw()