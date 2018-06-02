import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense,Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np

#文件夹预先分好train和test，其中，每个文件下放要分的不同的类，一类一个文件夹
#如/root/桌面/resnet尝试/train/1/*.jpg
#如/root/桌面/resnet尝试/train/2/*.png
DATA_DIR = '/media/root/72572323-0458-4f59-9fd6-33d6839809a0/胃神经瘤/data_new/'
OUTPUT_PATH='/media/root/72572323-0458-4f59-9fd6-33d6839809a0/胃神经瘤/data_new/model/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'test')
SIZE = (224,480)
BATCH_SIZE = 32


if __name__ == "__main__":

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    # val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    #换不同的分类网络就行了，备选有:
    # VGG16,VGG19,ResNet50,InceptionV3,InceptionResNetV2,Xception,MobileNet,DenseNet121, DenseNet169, DenseNet201,NASNetMobile, NASNetLarge
    #例vgg6调用格式为：keras.applications.vgg16.VGG16()，具体可以右键applications→go to→declaration
    model = keras.applications.vgg19.VGG19(weights=None,input_shape=(224,480,3),include_top=False)
    # model = keras.applications.resnet50.ResNet50(weights=None,input_shape=(1920,896,3),include_top=False)
    # model = keras.applications.inception_resnet_v2.InceptionResNetV2()
    num=0
    classes = list(iter(batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=True
    last = model.layers[-1].output
    x = Flatten(name='flatten')(last)
    x = Dense(60, activation='relu', name='fc1')(x)
    x = Dense(60, activation='relu', name='fc2')(x)
    x = Dense(len(classes), activation="softmax")(x)

    # x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    finetuned_model.summary()
    finetuned_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=20)#当训练patience个epoch,测试集的loss还没有下降时停止训练
    checkpointer = ModelCheckpoint(OUTPUT_PATH+'epoch{epoch:03d}_acc{val_acc:.3f}.h5', verbose=1, save_best_only=True)

    finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=1000, callbacks=[early_stopping, checkpointer],
                                  validation_data=val_batches, validation_steps=num_valid_steps)#callbacks=[early_stopping, checkpointer]
    finetuned_model.save(OUTPUT_PATH+'final_epoch{epoch:03d}_acc{val_acc:.3f}.h5')
