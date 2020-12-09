import os
import glob
import numpy as np

import tensorflow as tf
from tensorflow import keras

from gan import Generator, Discriminator
from data import make_anime_dataset

from PIL import Image
import matplotlib.pyplot as plt

# 梯度惩罚项计算函数
def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]
    # 每个样本均随机采样t,用于插值
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # 自动扩展为x 的形状，[b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)
    # 在真假图片之间做线性插值
    interplate = t * batch_x + (1 - t) * fake_image
    # 在梯度环境中计算D 对插值样本的梯度
    with tf.GradientTape() as tape:
        tape.watch([interplate])  # 加入梯度观察列表
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # 计算每个样本的梯度的范数:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    # 计算梯度惩罚项
    gp = tf.reduce_mean((gp - 1.) ** 2)
    return gp

# 判别器的损失函数
def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    fake_image = generator(batch_z, is_training) # 生成器生成样本
    d_fake_logits = discriminator(fake_image, is_training) # 判别器识别生成器样本
    d_real_logits = discriminator(batch_x, is_training) # 判别器识别真样本
    # 计算梯度惩罚项
    gp = gradient_penalty(discriminator, batch_x, fake_image)
    # WGAN-GP D 损失函数的定义，这里并不是计算交叉熵，而是直接最大化正样本的输出
    # 最小化假样本的输出和梯度惩罚项
    loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 10. * gp
    return loss, gp

# 生成器的损失函数
def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    # WGAN-GP G 损失函数，最大化假样本的输出值
    loss = - tf.reduce_mean(d_fake_logits)
    return loss

# 保存结果图片
def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])

    for b in range(val_out.shape[0]):
        # 每行的图片
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)
        # 每行的图片连接成整张图片
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    im = Image.fromarray(final_image)
    im.save('Anime_wgan_result.png')


d_losses, g_losses = [], []

# loss曲线图
def draw():
    plt.figure()
    plt.plot(g_losses, 'b', label='generator')
    plt.plot(d_losses, 'r', label='discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig('Anime_wgan_ACC.png')
    plt.show()


def main():
    batch_size = 512
    learning_rate = 0.002
    z_dim = 100
    is_training = True
    epochs = 300

    img_path = glob.glob(r'data/*.png')
    print('images num:', len(img_path))  # 输出图片的数量
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)  # (512, 64, 64, 3) (64, 64, 3)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())  # (512, 64, 64, 3) 1.0 -1.0
    dataset = dataset.repeat(100)  # 重复循环
    db_iter = iter(dataset)

    generator = Generator()  # 创建生成器
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()  # 创建判别器
    discriminator.build(input_shape=(None, 64, 64, 3))

    # 分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    # 首先从先验分布Pz中随机采样隐藏向量，从真实数据集中随机采样真实图片
    # 通过生成器和判别器计算判别网络的损失，并优化判别器网络参数theta
    # 在训练生成器时，需要借助于判别器来计算误差，但是只计算生成器的梯度信息，并更新fi
    # 判别器训练5次之后，生成器训练一次
    for epoch in range(epochs):
        # 1. 训练判别器:
        for _ in range(5):
            batch_z = tf.random.normal([batch_size, z_dim]) # 采样隐藏向量
            batch_x = next(db_iter) # 采集真实图片
            # 判别器向前计算
            with tf.GradientTape() as tape:
                # 判别器损失
                d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            # 判别器梯度
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            # 判别器优化
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # 2. 训练生成器:
        batch_z = tf.random.normal([batch_size, z_dim]) # 采样隐藏向量
        batch_x = next(db_iter) # 采样真实图片
        # 生成器向前计算
        with tf.GradientTape() as tape:
            # 生成器损失
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        # 生成器梯度
        grads = tape.gradient(g_loss, generator.trainable_variables)
        # 优化生成器
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        
        d_losses.append(float(d_loss))
        g_losses.append(float(g_loss))

        # 阶段性显示loss
        if epoch % 10 == 0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss), 'gp:', float(gp))
            z = tf.random.uniform([100, z_dim])

            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'wgan-%d.png'%epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')


if __name__ == '__main__':
    main()
    draw()