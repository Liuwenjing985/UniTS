import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from aeon.datasets import  write_to_tsfile

# 定义输入和输出文件夹
input_folder = 'dataset/intel'
output_folder = 'dataset/converted_data'
os.makedirs(output_folder, exist_ok=True)

# 初始化数据列表
data_list =[]
labels = np.array([])

# 遍历输入文件夹中的所有 .mat 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.mat'):
        try:
            # 读取 .mat 文件
            mat_data = scipy.io.loadmat(os.path.join(input_folder, filename))
            # 添加标签（文件名去掉扩展名）
            label = os.path.splitext(filename)[0]
             # 假设数据存储在 'data' 键中，您需要根据实际情况调整
            data = mat_data[label]  # 替换为实际的键名
            real_part = np.real(data)
            imag_part = np.imag(data)

            result = np.stack((real_part,imag_part),axis = 1)         
            data_list.append(result)
            label_full = np.full(data.shape[0],label)
            labels = np.concatenate((labels,label_full))
        except Exception as e:
            print(f"无法读取文件 {filename}: {e}")
            continue  # 跳过无法读取的文件
data_list = np.vstack(data_list)

train_data,test_data,train_labels,test_labels = train_test_split(data_list,labels,test_size=0.2,random_state = 42)


# 保存训练集和测试集为 .ts 文件
ts_file_path = 'dataset/converted_data/'
write_to_tsfile(train_data,ts_file_path,train_labels,problem_name="newdata_TRAIN.ts")
write_to_tsfile(test_data,ts_file_path,test_labels,problem_name="newdata_TEST.ts") 


print(f"Training data saved to {ts_file_path}")
print("运行完毕")


