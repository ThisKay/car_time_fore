import h5py
import pandas
import tensorflow as tf
import numpy as np
import confUtil as confu

config = confu.SmallConfig()
FEATURE_NUM = config.vocab_size

MIN_X = 0
MAX_X = 0
MIN_Y = 0
MAX_Y = 0
SCALER_XS = None
SCALER_YS = None


def read_traj_data(_path, _day_or_slot):
    f = h5py.File(_path, 'r')
    split = -468
    xs = f['xs'][:]
    ys = f['ys'][:]
    p_train = (xs.copy()[:split], ys.copy()[:split])
    p_test = (xs.copy()[split:], ys.copy()[split:])
    p_all = (xs.copy()[:], ys.copy()[:])
    print("[Log] read_data train:", p_train[0].shape)
    print("[Log] read_data test:", p_test[0].shape)
    f.close()
    return p_train, p_test, p_all, FEATURE_NUM


def ptb_producer_by_slot(raw_data, batch_size, num_steps, epoch_size, stride, name=None):
    '''
    将raw_data转换为batch_data
    '''
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        # 1、读入数据
        # (num,time_steps,3200)
        raw_data_x, raw_data_y = raw_data
        raw_data_x = tf.convert_to_tensor(raw_data_x, name="raw_data_x", dtype=tf.float32)
        raw_data_y = tf.convert_to_tensor(raw_data_y, name="raw_data_y", dtype=tf.float32)
        data_len = tf.shape(raw_data_x)[0]
        # 2、转换batch
        # (batch_size,batch_len,time_steps,3200)
        batch_len = data_len // batch_size
        epoch_size = batch_len
        data_x = tf.reshape(
            raw_data_x[0: batch_size * batch_len], [batch_size, batch_len, num_steps, FEATURE_NUM])
        data_y = tf.reshape(
            raw_data_y[0: batch_size * batch_len], [batch_size, batch_len, num_steps, FEATURE_NUM])

        # 3、assert
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        # 4、切片返回
        # (batch_size,batch_len,time_steps,3200)
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data_x, [0, i, 0, 0], [batch_size, i + 1, num_steps, FEATURE_NUM])
        y = tf.strided_slice(data_y, [0, i, 0, 0], [batch_size, i + 1, num_steps, FEATURE_NUM])
        # (batch_size,time_steps,3200)
        x = tf.reshape(x, [batch_size, num_steps, FEATURE_NUM])
        y = tf.reshape(y, [batch_size, num_steps, FEATURE_NUM])
        x.set_shape([batch_size, num_steps, FEATURE_NUM])
        y.set_shape([batch_size, num_steps, FEATURE_NUM])
        return x, y
