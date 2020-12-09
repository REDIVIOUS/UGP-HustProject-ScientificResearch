# import csv

# daterange = [31,28,31,30,31,30,31,31,30,31,30,31]
# for j in range(12):
#     for i in range(daterange[j]):
#         if i + 1 < 10:
#             if j + 1 < 10:
#                 month = '0'+str(j+1)
#             else:
#                 month = str(j+1)
#             path = 'data/'+str(j+1)+'/2017-'+month+'-'+'0'+str(i+1)+'.csv'
#             path0 = 'data/'+str(j+1)+'/big/2017-'+month+'-'+'0'+str(i+1)+'.csv'
#             path1 = 'data/'+str(j+1)+'/small/2017-'+month+'-'+'0'+str(i+1)+'.csv'
#         else:
#             if j + 1 < 10:
#                 month = '0'+str(j+1)
#             else:
#                 month = str(j+1)
#             path = 'data/'+str(j+1)+'/2017-'+month+'-'+str(i+1)+'.csv'
#             path0 = 'data/'+str(j+1)+'/big/2017-'+month+'-'+str(i+1)+'.csv'
#             path1 = 'data/'+str(j+1)+'/small/2017-'+month+'-'+str(i+1)+'.csv'
#         fcsv1 = open(path0, "w")
#         fcsv1.write("model, serial_number, failure, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\n")
#         fcsv2 = open(path1, "w")
#         fcsv2.write("model, serial_number, failure, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\n")
#         with open(path,'r') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 if row['model'] == 'ST4000DM000':
#                     fcsv1.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(row['model'],row['serial_number'],row['failure'],row['smart_1_normalized'],row['smart_4_raw'],row['smart_5_raw'],row['smart_7_normalized'],row['smart_9_normalized'],row['smart_10_normalized'],row['smart_12_raw'],row['smart_187_normalized'],row['smart_194_normalized'],row['smart_197_raw'],row['smart_198_raw'],row['smart_199_raw']))
#                 if row['model'] == 'ST8000DM002':
#                     fcsv2.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(row['model'],row['serial_number'],row['failure'],row['smart_1_normalized'],row['smart_4_raw'],row['smart_5_raw'],row['smart_7_normalized'],row['smart_9_normalized'],row['smart_10_normalized'],row['smart_12_raw'],row['smart_187_normalized'],row['smart_194_normalized'],row['smart_197_raw'],row['smart_198_raw'],row['smart_199_raw']))



import csv
import os

# daterange = [31,28,31,30,31,30,31,31,30,31,30,31]
# for j in range(12):
#     if j + 1 < 10:
#         month = '0' + str(j+1)
#     else:
#         month = str(j+1) 
#     pathori1 = 'data/'+str(j+1)+'/big/2017-'+month+'-01.csv'
#     pathori2 = 'data/'+str(j+1)+'/small/2017-'+month+'-01.csv'
#     with open(pathori1, 'r') as f:
#         reader = csv.DictReader(f)
#         disk = [row[' serial_number'] for row in reader]
#         for k in disk:
#             path = 'fdata/'+str(j+1)+'/bigdisk/'+k+'.csv'
#             fcsv = open(path, 'w')
#             fcsv.write("1,2,3,4,5,6,7,8,9,10,11,12,fail\n")
#     with open(pathori2, 'r') as f:
#         reader = csv.DictReader(f)
#         disk = [row[' serial_number'] for row in reader]
#         for k in disk:
#             path = 'fdata/'+str(j+1)+'/smalldisk/'+k+'.csv'
#             fcsv = open(path, 'w')
#             fcsv.write("1,2,3,4,5,6,7,8,9,10,11,12,fail\n")

#     filelist1 = os.listdir('fdata/'+str(j+1)+'/bigdisk')
#     filelist2 = os.listdir('fdata/'+str(j+1)+'/smalldisk')
    

#     for i in range(daterange[j]):
#         if i+1<10:
#             date = '0'+str(i+1)
#         else:
#             date = str(i+1)
#         path = 'data/'+str(j+1)+'/big/2017-'+month+'-'+date+'.csv'
#         with open(path, 'r') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 path0 = 'fdata/'+str(j+1)+'/bigdisk/'+row[' serial_number']+'.csv'
#                 if row[' serial_number']+'.csv' in filelist1:
#                     fcsv = open(path0, 'a')
#                     fcsv.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(row[' 1'],row[' 2'],row[' 3'],row[' 4'],row[' 5'],row[' 6'],row[' 7'],row[' 8'],row[' 9'],row[' 10'],row[' 11'],row[' 12'],row[' failure']))
            
#         path = 'data/'+str(j+1)+'/small/2017-'+month+'-'+date+'.csv'
#         with open(path, 'r') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 path0 = 'fdata/'+str(j+1)+'/smalldisk/'+row[' serial_number']+'.csv'
#                 if row[' serial_number']+'.csv' in filelist2:
#                     fcsv = open(path0, 'a')
#                     fcsv.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(row[' 1'],row[' 2'],row[' 3'],row[' 4'],row[' 5'],row[' 6'],row[' 7'],row[' 8'],row[' 9'],row[' 10'],row[' 11'],row[' 12'],row[' failure']))



import os
import csv
import numpy as np
from collections import deque


# # 参数
# # T: interval
# # option: bigdisk(1) or smalldisk(0)
# # month: which month to choose
# def OneDtoTwoD(T, option, month):
#     # 返回值
#     y_normal = [] # 未失效lable
#     x_normal = [] # 未失效特征矩阵

#     x_abnormal = [] # 失效特征矩阵
#     y_abnormal = [] # 失效label


#     # path代表当前的月份以及选择大磁盘或者是小磁盘
#     if option == 1:
#         disk = 'bigdisk'
#     else:
#         disk = 'smalldisk'
#     path = 'fdata/'+str(month)+'/'+disk
#     # filelist用于遍历该文件夹下所有的文件
#     filelist = os.listdir(path)
#     for file in filelist:
#         # 对每一个文件作处理
#         with open(path+'/'+file,'r') as f:
#             reader = csv.reader(f)
#             d = deque(maxlen = T)
#             lens = 0
#             flag = 0
#             for line in reader:
#                 if lens == 0:
#                     lens = lens + 1
#                     continue
#                 if lens < T:
#                     lens = lens + 1
#                     d.append(line[0:12])
#                     flag = flag or int(line[12])
#                 else:
#                     lens = lens + 1
#                     d.append(line[0:12])
#                     flag = flag or int(line[12])
#                     if flag == 1:
#                         x_abnormal.append(np.transpose(d))
#                         y_abnormal.append(flag)
#                     else:
#                         x_normal.append(np.transpose(d))
#                         y_normal.append(flag)
#     x_abnormal = np.array(x_abnormal, np.float32)
#     y_abnormal = np.array(y_abnormal, np.int)
#     x_normal = np.array(x_normal, np.float32)
#     y_normal = np.array(y_normal, np.int)
#     return x_abnormal, y_abnormal, x_normal, y_normal





# 参数
# T: interval
# option: bigdisk(1) or smalldisk(0)
# month: which month to choose
def OneDtoTwoD(T, option, month):
    # 返回值
    y_normal = [] # 未失效lable
    x_normal = [] # 未失效特征矩阵

    x_abnormal = [] # 失效特征矩阵
    y_abnormal = [] # 失效label

    # path代表当前的月份以及选择大磁盘或者是小磁盘
    if option == 1:
        disk = 'bigdisk'
    else:
        disk = 'smalldisk'
    path = 'fdata/'+str(month)+'/'+disk

    # filelist用于遍历该文件夹下所有的文件
    filelist = os.listdir(path)
    for file in filelist:
        with open(path+'/'+file,'r') as f:
            raw_lines = f.read().splitlines()
            cur_x = []
            cur_y = []
            lens = len(raw_lines)

            if lens >= T + 1:
                for i in range(lens-T,lens):
                    cur = raw_lines[i][0:len(raw_lines[i])-2].split(',')
                    cur = list(map(np.float32,cur))
                    cur_x.append(cur)
                cur_y = int(raw_lines[-1][-1])
                if(cur_y == 1):
                    x_abnormal.append(cur_x)
                    y_abnormal.append(cur_y)
                else:
                    x_normal.append(cur_x)
                    y_normal.append(cur_y)

    x_abnormal = np.array(x_abnormal, np.float32)
    y_abnormal = np.array(y_abnormal, np.int)
    x_normal = np.array(x_normal, np.float32)
    y_normal = np.array(y_normal, np.int)
    return x_abnormal, y_abnormal, x_normal, y_normal

