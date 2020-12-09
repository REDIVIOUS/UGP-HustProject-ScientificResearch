import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, Model, Sequential

from PIL import Image
import matplotlib.pyplot as plt


batchsz = 128 # 批量大小
h_dim = 20 # 中间隐藏层维度
lr = 0.001 # 学习率

# 加载Fashion MNIST图片数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)

# 归一化
x_train, x_test = x_train.astype(np.float32)/255., x_test.astype(np.float32)/255.
# 通过图片数据即可构建数据集对象，不需要标签
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(10000).batch(batchsz)
# 构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.shuffle(1000).batch(batchsz)

# 自编码器类，包含编码器和解码器
class AE(Model):
    def __init__(self):
        super(AE, self).__init__()
        # 创建编码网络
        self.encoder = Sequential([
            layers.Dense(256, activation = tf.nn.relu),
            layers.Dense(128, activation = tf.nn.relu),
            layers.Dense(h_dim)])
        # 创建解码网络
        self.decoder = Sequential([ 
            layers.Dense(128, activation = tf.nn.relu),
            layers.Dense(256, activation = tf.nn.relu),
            layers.Dense(784)])

    # 向前传播函数
    def call(self, inputs, training = None):
        # 编码获得隐藏向量h, [b, 784] => [b, 20]
        h = self.encoder(inputs)
        # 解码获得重建图片, [b, 20] => [b,784]
        x_hat = self.decoder(h)
        return x_hat

# 创建280x280大小图片阵列
def save_images(imgs, name):
    new_im = Image.new('L', (280,280))
    index = 0
    for i in range(0,280,28): # 10行图片阵列
        for j in range(0,280,28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
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
    plt.savefig('AE_ACC.png')
    plt.show()

# 创建网络对象
model = AE()
# 指定输入大小
model.build(input_shape=(None, 784))
# 打印网络信息
model.summary()
# 创建优化器，并设置学习率
optimizer = optimizers.Adam(lr=lr)
# 保存训练和测试过程中的误差
train_tot_loss = []
test_tot_loss = []

def main():
    for epoch in range(200): # 训练200个Epoch
        # 下面是训练过程
        cor, tot = 0, 0
        for step, x in enumerate(train_db): # 遍历训练集
            # 打平[b, 28, 28] => [b, 784]
            x = tf.reshape(x,[-1, 784])
            # 构造梯度记录器
            with tf.GradientTape() as tape:
                # 前向计算获得重建的图片
                x_rec_logits = model(x)
                # 计算重建图片与输入之间的损失函数
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = x, logits = x_rec_logits)
                #计算均值
                rec_loss = tf.reduce_mean(rec_loss)
                cor += rec_loss
                tot += x.shape[0]
                # 自动求导，包含了两个子网络的梯度
                grads = tape.gradient(rec_loss, model.trainable_variables)
                # 自动更新，同时更新两个子网络
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                # 间隔性打印训练误差
                print(epoch, step, float(rec_loss))
        train_tot_loss.append(cor/tot)

        # 下面是测试过程
        correct, total = 0,0
        for x in test_db:
            x = tf.reshape(x, [-1, 784])
            out = model(x)
            out_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = x, logits=out)
            # 计算均值
            loss = tf.reduce_mean(out_loss)
            correct += loss
            total += x.shape[0]
        test_tot_loss.append(correct/total)

        # 分别输出第1次、第10次和第100次的训练结果
        if(epoch == 0) or (epoch == 9) or (epoch == 99) or (epoch == 199):
            # 重建图像
            # 重建图片，从测试集采样一批图片
            x = next(iter(test_db))
            out_logits = model(tf.reshape(x, [-1, 784])) # 打平并送入自编码器
            x_hat = tf.sigmoid(out_logits) # 将输出转换成像素值，使用sigmoid函数
            # 恢复为28x28, [b,784] => [b,28,28]
            x_hat = tf.reshape(x_hat, [-1, 28, 28])
            # 输入的前50张+重建的前50张图片合并 （即左边为真实的图片，右边为重建图片，形成对比）
            x_concat = tf.concat([x[:50], x_hat[:50]],axis = 0)
            x_concat = x_concat.numpy() * 255 # 恢复为0～255的范围
            x_concat = x_concat.astype(np.uint8) # 转换为整型
            save_images(x_concat, 'AE_reconstruct_result_epoch_%d.png' % (epoch + 1)) # 保存图片

if __name__ == '__main__':
    main()
    draw()
