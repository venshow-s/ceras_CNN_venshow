import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

# def __init__(self, img_rows=512, img_cols=512):
#     self.img_rows = img_rows
#     self.img_cols = img_cols


def precision(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    true_positives = keras.sum(keras.round(keras.clip(y_true_f * y_pred_f, 0, 1)))
    predicted_positives = keras.sum(keras.round(keras.clip(y_pred_f, 0, 1)))
    precision = true_positives / (predicted_positives + keras.epsilon())
    return precision

def tp(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    true_positives = keras.sum(keras.round(keras.clip(y_true_f * y_pred_f, 0, 1)))
    return true_positives

def fp(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    tp_f01 = keras.round(keras.clip(y_true_f * y_pred_f, 0, 1))
    false_positives = keras.sum(keras.round(keras.clip(y_pred_f01-tp_f01, 0, 1)))
    return false_positives

def tn(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    all_one = keras.ones_like(y_pred_f01)
    y_pred_f_1 = -1*(y_pred_f01-all_one)
    y_true_f_1 = -1*(y_true_f-all_one)
    true_negatives = keras.sum(keras.round(keras.clip(y_true_f_1+y_pred_f_1, 0, 1)))
    return true_negatives

def fn(y_true,y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    # y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    tp_f01 = keras.round(keras.clip(y_true_f * y_pred_f, 0, 1))
    false_negatives = keras.sum(keras.round(keras.clip(y_true_f-tp_f01, 0, 1)))
    return false_negatives

def recall(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    true_positives = keras.sum(keras.round(keras.clip(y_true_f * y_pred_f, 0, 1)))
    possible_positives = keras.sum(keras.round(keras.clip(y_true_f, 0, 1)))
    recall = true_positives / (possible_positives + keras.epsilon())
    return recall


def fmeasure(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    fmeasure = (2 * p * r) / (p + r)
    return fmeasure

def uNet():

    inputs = Input(shape=(512, 512, 3), name='input')

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    print("conv1 shape:", conv1.shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    print("conv1 shape:", conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("pool1 shape:", pool1.shape)


    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    print("conv2 shape:", conv2.shape)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    print("conv2 shape:", conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    print("pool2 shape:", pool2.shape)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    print("conv3 shape:", conv3.shape)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    print("conv3 shape:", conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))
    merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))
    merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))
    merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv8))
    merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-7), loss='binary_crossentropy', metrics=[tp, fp, tn, fn, 'acc', precision, recall, fmeasure])

    return model


# 读取h5数据
PathH5 = "/media/root/72572323-0458-4f59-9fd6-33d6839809a0/2_23unet分割/segment/hdf5/train/"
listFileH5 = []
model_suffix = ".h5"
Model_save_Path = '/media/root/72572323-0458-4f59-9fd6-33d6839809a0/2_23unet分割/segment/model/'


for root, sub_dirs, filelist in os.walk(PathH5):
    for filename in filelist:
        listFileH5.append(filename)

# print(listFileH5)
trainset_num = len(listFileH5)
data_c  = np.zeros([1, 512, 512, 3], dtype=float)
label_c = np.zeros([1, 512, 512, 1], dtype=float)


if __name__ == "__main__":

    model = uNet()
    # model.load_weights('/media/root/72572323-0458-4f59-9fd6-33d6839809a0/2_23unet分割/segment/model_aaa/299640.h5')
    print(model.summary())

    for num in range(0,3000000):

        f = h5py.File(PathH5+listFileH5[num%trainset_num])
        data  = f['data'][:]
        label = f['label'][:]
        data_c[0,:,:,:] = data[:,:,:]
        label_c[0,:,:,0] = label[:,:]
        result = model.train_on_batch(data_c,label_c)

        if num% (10*trainset_num) == 0:
            result_test = model.test_on_batch(data_c, label_c)
            tp = result_test[1]
            fp = result_test[2]
            tn = result_test[3]
            fn = result_test[4]
            acc=(tp + tn) / ((tp + tn + fp + fn)) * 100
            precision = (tp / (tp + fp)) * 100
            recall = (tp / (tp + fn)) * 100
            dice = (2*tp/(2*tp+fn+fp))*100
            str_num = str(num)
            model.save(Model_save_Path + str_num + model_suffix)
            print("iteration", str_num, " loss_seg:", result[0],' precision:',str(precision),' recall:',str(recall),' dice:',dice)
