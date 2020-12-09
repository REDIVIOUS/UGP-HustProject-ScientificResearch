import torch
import numpy as np
import tensorflow as tf
import processdata as pro
from sklearn.utils import shuffle

class Dataset(object):
    def __init__(self, T, option, month, normalize=False):

        # 开始载入数据
        print("\nInitializing Dataset...")
        self.normalize = normalize
        self.T = T
        self.option = option
        self.month = month

        # 载入mnist数据
        x_abnormal, y_abnormal, x_normal, y_normal = pro.OneDtoTwoD(T,option,month)
        print('data loading done, starting to split...\n')

        state = np.random.get_state()
        np.random.shuffle(x_normal)
        np.random.set_state(state)
        np.random.shuffle(y_normal)

        if len(x_normal) > 3000:
            x_normal = x_normal[0:3000]
            y_normal = y_normal[0:3000]

        lendata = len(x_normal)
        traning_size = int(0.7 * lendata)

        # 训练集，全是正常的
        self.x_tr, self.y_tr = x_normal[:traning_size], y_normal[:traning_size]

        # 测试集，有正常的也有异常的
        # 测试集，正常部分
        self.x_te, self.y_te = x_normal[traning_size:], y_normal[traning_size:]

        # 测试集，有异常的
        self.x_te = np.append(self.x_te, x_abnormal, axis=0)
        self.y_te = np.append(self.y_te, y_abnormal, axis=0)

        # 训练集数量和测试集数量
        self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
        self.idx_tr, self.idx_te = 0, 0

        # 输出训练集和测试集的数量
        print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

        # width和height的信息
        x_sample, y_sample = self.x_te[0], self.y_te[0]
        self.height = x_sample.shape[0]
        self.width = x_sample.shape[1]
        try: self.channel = x_sample.shape[2]
        except: self.channel = 1

        # 帮助显示信息
        self.min_val, self.max_val = x_sample.min(), x_sample.max()

        # 显示数据集信息
        print("Information of data")
        print("Shape  Height: %d, Width: %d, Channel: %d" %(self.height, self.width, self.channel))
        # 显示是否正则化
        print("Normalization: %r" %(self.normalize))
        if(self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" %(self.min_val, self.max_val, 0, 1))

    # 重置当前的训练集和测试集指针
    def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

    # 下一个训练batch
    def next_train(self, batch_size=1, fix=False):
        start, end = self.idx_tr, self.idx_tr+batch_size
        # 当前的训练batch范围
        x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]
        x_tr = np.expand_dims(x_tr, axis=3)

        terminator = False
        # 数据集取完
        if(end >= self.num_tr):
            terminator = True
            # 重新开始取
            self.idx_tr = 0
            self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        # 训练集合指针指向结束
        else: self.idx_tr = end

        # 固定从第一个元素开始
        if(fix): self.idx_tr = start

        # 如果最后一个bathc不足，则从最后一个元素开始往前取batch_size
        if(x_tr.shape[0] != batch_size):
            x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]
            x_tr = np.expand_dims(x_tr, axis=3)

        # 是否需要正则化
        if(self.normalize):
            min_x, max_x = x_tr.min(), x_tr.max()
            x_tr = (x_tr - min_x) / (max_x - min_x)

        # to_torch
        x_tr_torch = torch.from_numpy(np.transpose(x_tr, (0, 3, 1, 2)))
        y_tr_torch = torch.from_numpy(y_tr)

        return x_tr, x_tr_torch, y_tr, y_tr_torch, terminator

    # 下一个测试batch
    def next_test(self, batch_size=1):
        start, end = self.idx_te, self.idx_te+batch_size
        # 当前测试batch的范围
        x_te, y_te = self.x_te[start:end], self.y_te[start:end]
        x_te = np.expand_dims(x_te, axis=3)

        terminator = False
        # 最后一个bathc取完或者不足，即终止
        if(end >= self.num_te):
            terminator = True
            self.idx_te = 0
        # 测试集指针指向end
        else: self.idx_te = end

        # 是否需要正则化
        if(self.normalize):
            min_x, max_x = x_te.min(), x_te.max()
            x_te = (x_te - min_x) / (max_x - min_x)

        # to_torch
        x_te_torch = torch.from_numpy(np.transpose(x_te, (0, 3, 1, 2)))
        y_te_torch = torch.from_numpy(y_te)

        return x_te, x_te_torch, y_te, y_te_torch, terminator



# # mnist测试

# import torch
# import numpy as np
# import tensorflow as tf
# from sklearn.utils import shuffle

# class Dataset(object):
#     def __init__(self, normalize=True):

#         # 开始载入数据
#         print("\nInitializing Dataset...")
#         self.normalize = normalize

#         # 载入mnist数据
#         (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()

#         # 训练集图像、标签；测试集图像、标签
#         self.x_tr, self.y_tr = x_tr, y_tr
#         self.x_te, self.y_te = x_te, y_te

#         # 数据类型转换
#         self.x_tr = np.ndarray.astype(self.x_tr, np.float32)
#         self.x_te = np.ndarray.astype(self.x_te, np.float32)

#         # 分割数据集
#         self.split_dataset()
#         # 训练集数量和测试集数量
#         self.num_tr, self.num_te = self.x_tr.shape[0], self.x_te.shape[0]
#         self.idx_tr, self.idx_te = 0, 0

#         # 输出训练集和测试集的数量
#         print("Number of data\nTraining: %d, Test: %d\n" %(self.num_tr, self.num_te))

#         # width和height的信息
#         x_sample, y_sample = self.x_te[0], self.y_te[0]
#         self.height = x_sample.shape[0]
#         self.width = x_sample.shape[1]
#         try: self.channel = x_sample.shape[2]
#         except: self.channel = 1

#         # 帮助显示信息
#         self.min_val, self.max_val = x_sample.min(), x_sample.max()
#         self.num_class = (y_te.max()+1)

#         # 显示信息
#         print("Information of data")
#         print("Shape  Height: %d, Width: %d, Channel: %d" %(self.height, self.width, self.channel))
#         print("Value  Min: %.3f, Max: %.3f" %(self.min_val, self.max_val))
#         print("Class  %d" %(self.num_class))
#         print("Normalization: %r" %(self.normalize))
#         if(self.normalize): print("(from %.3f-%.3f to %.3f-%.3f)" %(self.min_val, self.max_val, 0, 1))

#     # 分割数据
#     def split_dataset(self):

#         # 将测试集和训练集的x特征混在一起
#         x_tot = np.append(self.x_tr, self.x_te, axis=0)
#         # 将测试集和训练集的y特征混在一起
#         y_tot = np.append(self.y_tr, self.y_te, axis=0)

#         # 正常情况的x和y
#         x_normal, y_normal = None, None
#         # 异常情况的x和y
#         x_abnormal, y_abnormal = None, None

#         # 分出normal的和abnormal的
#         for yidx, y in enumerate(y_tot):
#             x_tmp = np.expand_dims(x_tot[yidx], axis=0)
#             y_tmp = np.expand_dims(y_tot[yidx], axis=0)

#             # 正常情况下
#             if(y == 1):
#                 if(x_normal is None):
#                     x_normal = x_tmp
#                     y_normal = y_tmp
#                 else:
#                     x_normal = np.append(x_normal, x_tmp, axis=0)
#                     y_normal = np.append(y_normal, y_tmp, axis=0)

#             # 异常情况下
#             else:
#                 if(x_abnormal is None):
#                     x_abnormal = x_tmp
#                     y_abnormal = y_tmp
#                 else:
#                     if(x_abnormal.shape[0] < 1000):
#                         x_abnormal = np.append(x_abnormal, x_tmp, axis=0)
#                         y_abnormal = np.append(y_abnormal, y_tmp, axis=0)

#             if(not(x_normal is None) and not(x_abnormal is None)):
#                 if((x_normal.shape[0] >= 2000) and x_abnormal.shape[0] >= 1000): break

#         # 训练集，全是正常的
#         self.x_tr, self.y_tr = x_normal[:1000], y_normal[:1000]

#         # 测试集，有正常的也有异常的
#         self.x_te, self.y_te = x_normal[1000:], y_normal[1000:]

#         # 测试集，有异常的
#         self.x_te = np.append(self.x_te, x_abnormal, axis=0)
#         self.y_te = np.append(self.y_te, y_abnormal, axis=0)

#     # 重置当前的训练集和测试集指针
#     def reset_idx(self): self.idx_tr, self.idx_te = 0, 0

#     # 下一个训练batch
#     def next_train(self, batch_size=1, fix=False):
#         start, end = self.idx_tr, self.idx_tr+batch_size
#         # 当前的训练batch范围
#         x_tr, y_tr = self.x_tr[start:end], self.y_tr[start:end]
#         x_tr = np.expand_dims(x_tr, axis=3)

#         terminator = False
#         # 数据集取完
#         if(end >= self.num_tr):
#             terminator = True
#             # 重新开始取
#             self.idx_tr = 0
#             self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
#         # 训练集合指针指向结束
#         else: self.idx_tr = end

#         # 固定从第一个元素开始
#         if(fix): self.idx_tr = start

#         # 如果最后一个bathc不足，则从最后一个元素开始往前取batch_size
#         if(x_tr.shape[0] != batch_size):
#             x_tr, y_tr = self.x_tr[-1-batch_size:-1], self.y_tr[-1-batch_size:-1]
#             x_tr = np.expand_dims(x_tr, axis=3)

#         # 是否需要正则化
#         if(self.normalize):
#             min_x, max_x = x_tr.min(), x_tr.max()
#             x_tr = (x_tr - min_x) / (max_x - min_x)

#         # to_torch
#         x_tr_torch = torch.from_numpy(np.transpose(x_tr, (0, 3, 1, 2)))
#         y_tr_torch = torch.from_numpy(y_tr)

#         return x_tr, x_tr_torch, y_tr, y_tr_torch, terminator

#     # 下一个测试batch
#     def next_test(self, batch_size=1):
#         start, end = self.idx_te, self.idx_te+batch_size
#         # 当前测试batch的范围
#         x_te, y_te = self.x_te[start:end], self.y_te[start:end]
#         x_te = np.expand_dims(x_te, axis=3)

#         terminator = False
#         # 最后一个bathc取完或者不足，即终止
#         if(end >= self.num_te):
#             terminator = True
#             self.idx_te = 0
#         # 测试集指针指向end
#         else: self.idx_te = end

#         # 是否需要正则化
#         if(self.normalize):
#             min_x, max_x = x_te.min(), x_te.max()
#             x_te = (x_te - min_x) / (max_x - min_x)

#         # to_torch
#         x_te_torch = torch.from_numpy(np.transpose(x_te, (0, 3, 1, 2)))
#         y_te_torch = torch.from_numpy(y_te)

#         return x_te, x_te_torch, y_te, y_te_torch, terminator
