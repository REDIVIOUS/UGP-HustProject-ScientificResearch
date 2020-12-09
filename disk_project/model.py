import os, glob, inspect, time, math, torch

import numpy as np
import matplotlib.pyplot as plt
import loss as lfs

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/."

def save_graph(contents, xlabel, ylabel, savename):

    np.save(savename, np.asarray(contents))
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(contents, color='blue', linestyle="-", label="loss")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig("%s.png" %(savename))
    plt.close()


# 训练过程
def training(neuralnet, dataset, epochs, batch_size):
    # 显示当前训练的轮数
    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    iteration = 0
    # test_sq = 20
    # test_size = test_sq**2

    # 分别保存四个loss的值
    list_enc, list_con, list_adv, list_tot = [], [], [], []

    # 训练epoch次
    for epoch in range(epochs):

        while(True):
            x_tr, x_tr_torch, y_tr, y_tr_torch, terminator = dataset.next_train(batch_size)

            # 分别代表压缩特征z_code、重构特征x_hat、重构后的压缩特征z_code_hat
            z_code = neuralnet.encoder(x_tr_torch.to(neuralnet.device))
            x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
            z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))

            # 判别器判别
            dis_x, features_real = neuralnet.discriminator(x_tr_torch.to(neuralnet.device))
            dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

            # 计算loss
            l_tot, l_enc, l_con, l_adv = lfs.loss_ganomaly(z_code, z_code_hat, x_tr_torch, x_hat, dis_x, dis_x_hat, features_real, features_fake)

            # 用l_tot进行优化
            neuralnet.optimizer.zero_grad()
            l_tot.backward()
            neuralnet.optimizer.step()

            # 分别计算
            list_enc.append(l_enc)
            list_con.append(l_con)
            list_adv.append(l_adv)
            list_tot.append(l_tot)

            # 迭代次数加一
            iteration += 1
            if(terminator): break

        # 显示loss
        print("Epoch [%d / %d] (%d iteration)  Enc:%.3f, Con:%.3f, Adv:%.3f, Total:%.3f" %(epoch, epochs, iteration, l_enc, l_con, l_adv, l_tot))

        # 储存模型参数
        for idx_m, model in enumerate(neuralnet.models):
            torch.save(model.state_dict(), PACK_PATH+"/runs/params-%d" %(idx_m))

    # 保存loss图线
    save_graph(contents=list_enc, xlabel="Iteration", ylabel="Enc Error", savename="l_enc")
    save_graph(contents=list_con, xlabel="Iteration", ylabel="Con Error", savename="l_con")
    save_graph(contents=list_adv, xlabel="Iteration", ylabel="Adv Error", savename="l_adv")
    save_graph(contents=list_tot, xlabel="Iteration", ylabel="Total Loss", savename="l_tot")

# 训练阶段
def test(neuralnet, dataset):

    # 载入模型参数
    param_paths = glob.glob(os.path.join(PACK_PATH, "runs", "params*"))
    param_paths.sort()

    if(len(param_paths) > 0):
        for idx_p, param_path in enumerate(param_paths):
            print(PACK_PATH+"/runs/params-%d" %(idx_p))
            neuralnet.models[idx_p].load_state_dict(torch.load(PACK_PATH+"/runs/params-%d" %(idx_p)))
            neuralnet.models[idx_p].eval()

    y_true_tot = 0 # 实际失效磁盘数量
    y_false_tot = 0 # 实际未失效磁盘数量

    y_true_predict = 0 # 失效预测成功数量
    y_false_predict = 0 # 错误预测失效数量

    # 开始进行测试
    print("\nTest...")
    # 分别表示正常图片和非正常图片的得分，方便计算界限
    scores_normal, scores_abnormal = [], []

    # 正常图片和异常图片混合，计算界限
    while(True):
        x_te, x_te_torch, y_te, y_te_torch, terminator = dataset.next_test(1) 

        # 计算第一次压缩值、重建值、重建压缩值
        z_code = neuralnet.encoder(x_te_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))

        # 判别器判别
        dis_x, features_real = neuralnet.discriminator(x_te_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))
        
        # 计算各个loss
        l_tot, l_enc, l_con, l_adv = lfs.loss_ganomaly(z_code, z_code_hat, x_te_torch, x_hat, dis_x, dis_x_hat, features_real, features_fake)
        # 用l_con来计算异常差异
        score_anomaly = l_enc.item()

        # if y_te == 1:
        #     scores_normal.append(score_anomaly)
        #     y_false_tot = y_false_tot + 1
        # else:
        #     scores_abnormal.append(score_anomaly)
        #     y_true_tot = y_true_tot + 1

        
        # 如果判定标签为1，即为磁盘失效
        if(y_te == 1):
            scores_abnormal.append(score_anomaly)
            y_true_tot = y_true_tot + 1
        # 如果判定标签不为1，即为异常
        else:
            scores_normal.append(score_anomaly)
            y_false_tot = y_false_tot + 1

        if(terminator): break

    # 根据得分计算出outbound
    scores_normal = np.asarray(scores_normal)
    scores_abnormal = np.asarray(scores_abnormal)
    normal_avg, normal_std = np.average(scores_normal), np.std(scores_normal)
    abnormal_avg, abnormal_std = np.average(scores_abnormal), np.std(scores_abnormal)
    print("Noraml  avg: %.5f, std: %.5f" %(normal_avg, normal_std))
    print("Abnoraml  avg: %.5f, std: %.5f" %(abnormal_avg, abnormal_std))
    outbound = normal_avg + (normal_std * 1.5)
    print("Outlier boundary of normal data: %.5f" %(outbound))

    # 正常图片和异常图片混合，用于测试
    # 记录结果
    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0

    # 正常图片和异常图片混合，开始测试
    while(True):
        x_te, x_te_torch, y_te, y_te_torch, terminator = dataset.next_test(1)

        # 计算第一次压缩值、重建值、重建压缩值
        z_code = neuralnet.encoder(x_te_torch.to(neuralnet.device))
        x_hat = neuralnet.decoder(z_code.to(neuralnet.device))
        z_code_hat = neuralnet.encoder(x_hat.to(neuralnet.device))

        # 判别器判别
        dis_x, features_real = neuralnet.discriminator(x_te_torch.to(neuralnet.device))
        dis_x_hat, features_fake = neuralnet.discriminator(x_hat.to(neuralnet.device))

        # 计算各个loss
        l_tot, l_enc, l_con, l_adv = lfs.loss_ganomaly(z_code, z_code_hat, x_te_torch, x_hat, dis_x, dis_x_hat, features_real, features_fake)
        # 用l_con来计算异常差异
        score_anomaly = l_enc.item()

        # 如果结果大于outbound，则异常
        outcheck = score_anomaly > outbound

        # # mnist 测试
        # if (y_te != 1 and outcheck == True):
        #     y_true_predict = y_true_predict + 1
        # if (y_te == 1 and outcheck == True):
        #     y_false_predict = y_false_predict + 1
        
        # 磁盘 测试
        if (y_te == 1 and outcheck == True):
            y_true_predict = y_true_predict + 1
        if (y_te == 0 and outcheck == True):
            y_false_predict = y_false_predict + 1

        # 判断结果和真是结果写入
        fcsv.write("%d, %.3f, %r\n" %(y_te, score_anomaly, outcheck))

        testnum += 1
        if(terminator): break
    
    # 计算FAR和FDR
    FDR = y_true_predict * 1.0 / y_true_tot
    FAR = y_false_predict * 1.0 / y_false_tot

    print('\n\nFAR = %f'%(FAR))
    print('\nFDR = %f'%(FDR))
    