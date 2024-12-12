import os
import aeon
import sys 

from aeon.datasets import load_from_ts_file


DATA_PATH = os.path.join(os.getcwd(), "dataset/converted_data")
print("查看一下data_path",DATA_PATH )

train_x, train_y = load_from_ts_file(DATA_PATH + "/newdata_TEST.ts")
test_x, test_y = load_from_ts_file(DATA_PATH + "/newdata_TRAIN.ts")
print("查看加载的数据",test_x.shape)
