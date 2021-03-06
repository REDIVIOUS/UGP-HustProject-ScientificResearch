import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch

import gan as nn
import data as dman
import model as solver

def main():
    # 取数据
    dataset = dman.Dataset(T=12, option=1, month=1, normalize=FLAGS.datnorm)
    # dataset = dman.Dataset(normalize=FLAGS.datnorm)
    # 探测GPU
    if(not(torch.cuda.is_available())): FLAGS.ngpu = 0
    device = torch.device("cuda" if (torch.cuda.is_available() and FLAGS.ngpu > 0) else "cpu")
    # 构建模型
    neuralnet = nn.Gan(height=dataset.height, width=dataset.width, channel=dataset.channel, device=device, ngpu=FLAGS.ngpu, ksize=FLAGS.ksize, z_dim=FLAGS.z_dim, learning_rate=FLAGS.lr)
    # 训练模型
    solver.training(neuralnet=neuralnet, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    # 预测模型
    solver.test(neuralnet=neuralnet, dataset=dataset)

if __name__ == '__main__':
    # 参数表
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int, default=12, help='the 2D trunk interval')
    parser.add_argument('--option', type=int, default=1, help='1=bigdisk, 2=smalldisk')
    parser.add_argument('--ngpu', type=int, default=1, help='-')
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--ksize', type=int, default=3, help='kernel size for constructing Neural Network')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of latent vector')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--batch', type=int, default=256, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()