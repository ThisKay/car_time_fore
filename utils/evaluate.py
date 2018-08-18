import os
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import h5py

_OUT_DIR = "../out/"
_DATA_DIR = "../dataset/"


def h5open(_file):
    '''
    读取h5py文件中的数据
    '''
    f = h5py.File(_file, 'r')
    _data = f['data'][:]
    f.close()
    return _data


def h5open_dict(_file):
    '''
    读取h5py文件中的字典数据并返回
    '''
    _dict = {}
    f = h5py.File(_file, 'r')
    for k, v in f.items():
        _dict[k] = v[:]
    f.close()
    return _dict


def calc_MARE(label, pred):
    '''
    平均绝对值相对误差
    为零的元素不计入累加值，不计入总数值
    '''
    abs_diff = np.abs(pred - label)
    divide = label.copy()
    divide = np.abs(divide)
    divide[label == 0] = 1  # 后面计算平均不会用到
    error = abs_diff / divide
    # print(lossTest.shape)  # (1, 20, 20, 8)
    # print(batch_ys.shape)  # (1, 8, 20, 20)
    error[error > 4] = 0
    error_ratio = error[label != 0].mean()
    return error_ratio


def calc_MSE(label, pred):
    '''
    平均绝对值相对误差
    为零的元素不计入累加值，不计入总数值
    '''
    mse = np.mean(np.square(pred - label))
    return mse


def output_compare(f_label, f_pred):
    '''将输出文件整理成为可直接比对的矩阵：1、reshape形状，2、extend范围
    '''
    grid_size = 20
    split = -360  # -360  # 630
    time_steps = 4  # 4
    label_data = h5open_dict(
        os.path.join(_DATA_DIR, f_label))['ys'].reshape([-1, time_steps, grid_size, grid_size, 8])
    label_data = label_data[:, -1]  # 取最后一个时间片
    label_data = label_data.reshape([-1, grid_size, grid_size, 8])

    label_data_ext = np.zeros([len(label_data), 20, 20, 8])
    label_data_ext[:, :grid_size, -grid_size:, :] = label_data
    label_data = label_data_ext
    label_data = label_data[split:]
    pred_data = h5open(os.path.join(_OUT_DIR, f_pred)).reshape([-1, grid_size, grid_size, 8])
    pred_data = pred_data[split:]
    pred_data_ext = label_data.copy()
    print(pred_data.shape, label_data.shape)

    pred_data_ext[:, : grid_size, -grid_size:, :] = pred_data
    assert pred_data_ext.shape == pred_data_ext.shape, "{},{}".format(label_data.shape, pred_data_ext.shape)
    # return label_data, pred_data_ext

    MARE_lst = []  # mean absolute releative error
    MSE_lst = []
    for idx in range(pred_data_ext.shape[0]):
        i = (idx) // 36 + 6  # day
        j = (idx) % 36 + 24  # slot
        mare = calc_MARE(label_data[idx], pred_data_ext[idx])
        mse = calc_MSE(label_data[idx], pred_data_ext[idx])  # / 121 * 400
        MARE_lst.append(mare)
        MSE_lst.append(mse)
        # print("d: %02d,s: %02d,are: %.2f,se: %.2f" % (i + 1, j, mare, mse))

    print(np.mean(MARE_lst))
    print(np.mean(MSE_lst))
    plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.set_xticks(np.arange(0, 648, 20))
    ax2.set_xticks(np.arange(0, 648, 20))
    ax1.plot(MARE_lst)
    ax2.plot(MSE_lst)
    plt.show()


if __name__ == '__main__':
    output_compare(f_label="train_by_day.h5", f_pred="out_by_day_e120.h5")
