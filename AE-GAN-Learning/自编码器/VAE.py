import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, Model, Sequential
from PIL import Image
import matplotlib.pyplot as plt

batchsz = 128  # 批量大小
lr = 0.001 # 学习率
z_dim = 20 # z的维度

# 加载Fashion MNIST 图片数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

# 归一化
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# 只需要通过图片数据即可构建数据集对象，不需要标签
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(10000).batch(batchsz)
# 构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.shuffle(1000).batch(batchsz)


class VAE(keras.Model):
    # 变分自编码器
    def __init__(self):
        super(VAE, self).__init__()
        # 编码器网络
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)  # 均值输出
        self.fc3 = layers.Dense(z_dim)  # 方差输出
        # 解码器网络
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        # 获得编码器的均值和方差
        h = tf.nn.relu(self.fc1(x))
        # 均值向量
        mu = self.fc2(h)
        # 方差的log 向量
        log_var = self.fc3(h)
        return mu, log_var

    def decoder(self, z):
        # 根据隐藏变量z 生成图片数据
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        # 返回数据图片
        return out

    def call(self, inputs, training=None):
        # 前向计算
        # 编码器[b, 784] => [b, z_dim], [b, z_dim]
        mu, log_var = self.encoder(inputs)
        # 采样reparameterization trick
        z = self.reparameterize(mu, log_var)
        # 通过解码器生成
        x_hat = self.decoder(z)
        # 返回生成样本，及其均值与方差
        return x_hat, mu, log_var

    def reparameterize(self, mu, log_var):
        # reparameterize 技巧，从正态分布采样epsion
        eps = tf.random.normal(log_var.shape)
        # 计算标准差
        std = tf.exp(log_var) ** 0.5
        # reparameterize 技巧
        z = mu + std * eps
        return z


def save_images(imgs, name):
    # 创建280x280 大小图片阵列
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):  # 10 行图片阵列
        for j in range(0, 280, 28):  # 10 列图片阵列
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))  # 写入对应位置
            index += 1
    # 保存图片阵列
    new_im.save(name)


def draw():
    plt.figure()
    plt.plot(train_tot_loss, 'b', label='train')
    plt.plot(test_tot_loss, 'r', label='test')
    plt.xlabel('Epoch')
    plt.ylabel('ACC')
    plt.legend()
    plt.savefig('VAE_ACC.png')
    plt.show()


# 创建网络对象
model = VAE()
# 指定输入大小
model.build(input_shape=(4, 784))
# 打印网络信息
model.summary()
# 创建优化器，并设置学习率
optimizer = optimizers.Adam(lr=lr)
# 保存训练和测试过程中的误差情况
train_tot_loss = []
test_tot_loss = []


def main():
    for epoch in range(200):  # 训练200个Epoch
        # 处理训练集
        cor, tot = 0, 0
        for step, x in enumerate(train_db):  # 遍历训练集
            # 打平，[b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 784])
            # 构建梯度记录器
            with tf.GradientTape() as tape:
                # 前向计算
                x_rec_logits, mu, log_var = model(x)
                # 重建损失值计算
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits = x_rec_logits)
                rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
                # 计算KL 散度 N(mu, var) VS N(0, 1)
                kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
                kl_div = tf.reduce_sum(kl_div) / x.shape[0]
                # 合并误差项
                loss = rec_loss + 1. * kl_div
                cor += loss
                tot += x.shape[0]
                # 自动求导
                grads = tape.gradient(loss, model.trainable_variables)
                # 自动更新
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                # 间隔性打印训练误差
                print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))
        train_tot_loss.append(cor / tot)

        # 处理测试集
        correct, total = 0, 0
        for x in test_db:
            x = tf.reshape(x, [-1, 784])
            # 前向计算
            x_rec_logits, mu, log_var = model(x)
            # 重建损失值计算
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
            # 计算KL 散度 N(mu, var) VS N(0, 1)
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]
            # 合并误差项
            loss = rec_loss + 1. * kl_div

            correct += loss
            total += x.shape[0]
        test_tot_loss.append(correct / total)

        # 显示第1个、第10个、第100个、第200个epoch的结果
        if (epoch == 0) or (epoch == 9) or (epoch == 99) or (epoch == 199):
            # 测试生成效果，从正态分布随机采样z
            z = tf.random.normal((batchsz, z_dim))
            logits = model.decoder(z)  # 仅通过解码器生成图片
            x_hat = tf.sigmoid(logits)  # 转换为像素范围
            x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
            x_hat = x_hat.astype(np.uint8)
            save_images(x_hat, 'VAE_generated_result_epoch_%d.png' % (epoch+1))  # 保存生成图片

            # 重建图片，从测试集采样图片
            x = next(iter(test_db))
            logits, _, _ = model(tf.reshape(x, [-1, 784]))  # 打平并送入自编码器
            x_hat = tf.sigmoid(logits)  # 将输出转换为像素值
            # 恢复为28x28,[b, 784] => [b, 28, 28]
            x_hat = tf.reshape(x_hat, [-1, 28, 28])
            # 输入的前50 张+重建的前50 张图片合并 （即左边为真实的图片，右边为重建图片，形成对比）
            x_concat = tf.concat([x[:50], x_hat[:50]], axis=0)
            x_concat = x_concat.numpy() * 255.  # 恢复为0~255 范围
            x_concat = x_concat.astype(np.uint8)
            save_images(x_concat, 'VAE_reconstruct_result_epoch_%d.png' % (epoch+1))  # 保存重建图片


if __name__ == '__main__':
    main()
    draw()