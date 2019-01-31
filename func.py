import glob
import os

import cv2
import numpy as np
from keras import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.engine.saving import model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_preprocessing.image import load_img, img_to_array

from history_checkpoint_callback import HistoryCheckpoint

CLASSES = ["dog", "cat"]
IMAGE_SHAPE = (64, 64, 3)
TRAIN_DIR = "data/{}/train"
VALIDATE_DIR = "data/{}/valid"
TEST_DIR = "data/{}/test"


def load_images(dir_format):
    x = []
    y = []
    files = []
    for i, class_name in enumerate(CLASSES):
        f, images = load_images_by_dir(dir_format.format(class_name), IMAGE_SHAPE)
        files += f
        for image in images:
            x.append(image)
            y.append(i)
    y = np_utils.to_categorical(y, len(CLASSES))
    return np.array(x), y, files

def load_images_by_dir(dir_name, image_shape, with_normalize=True):
    files = glob.glob(os.path.join(dir_name, '*.jpg'))
    files.sort()

    images = []
    print('load_images : ', len(files))
    for i, file in enumerate(files):
        img = load_image(file, image_shape, with_normalize)
        images.append(img)
        if i % 500 == 0:
            print('load_images loaded ', i)
    return (files, np.array(images, dtype=np.float32))


def load_image(file_name, image_shape, with_normalize=True):
    src_img = cv2.imread(file_name)
    if src_img is None:
        return None

    dist_img = src_img
    # if not is_binary and image_shape[2] == 1:
    #     dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)

    dist_img = cv2.resize(dist_img, (image_shape[0], image_shape[1]))
    if with_normalize:
        dist_img = dist_img / 255
    return dist_img


################################
###### 画像データの前処理 ######
################################


def pre_process(dirname, filename, var_amount=3):
    num = 0
    arrlist = []
    files = glob.glob(dirname + "/*.jpg")

    for imgfile in files:
        img = load_img(imgfile, target_size=(hw["height"], hw["width"]))  # 画像ファイルの読み込み
        array = img_to_array(img) / 255  # 画像ファイルのnumpy化
        arrlist.append(array)  # numpy型データをリストに追加
        for i in range(var_amount - 1):
            arr2 = array
            arrlist.append(arr2)  # numpy型データをリストに追加
        num += 1

    nplist = np.array(arrlist)
    np.save(filename, nplist)
    print(">> " + dirname + "から" + str(num) + "個のファイル読み込み成功")


################################
######### モデルの構築 #########
################################

def build_cnn():
    IMAGE_SHAPE = (64, 64, 3)

    inputs = Input(IMAGE_SHAPE)

    cv1_1 = Conv2D(16, 3, padding='same', input_shape=IMAGE_SHAPE)(inputs)
    cv1_1 = Activation('relu')(cv1_1)

    cv1_2 = Conv2D(16, 3, padding='same')(cv1_1)
    cv1_2 = Activation('relu')(cv1_2)
    cv1_2 = Dropout(0.5)(cv1_2)

    mp1 = MaxPooling2D(2)(cv1_2)

    # cv2_1 = Conv2D(32, 3, padding='same')(mp1)
    # cv2_1 = Activation('relu')(cv2_1)
    # cv2_2 = Conv2D(32, 3, padding='same')(cv2_1)
    # cv2_2 = Activation('relu')(cv2_2)
    # cv2_2 = Dropout(0.5)(cv2_2)
    # mp2 = MaxPooling2D(2)(cv2_2)

    fl = Flatten()(mp1)

    # fc1 = Dense(1000)(fl)
    # fc1 = Activation('relu')(fc1)

    fc1 = Dropout(0.5)(fl)

    outputs = Dense(2)(fc1)
    outputs = Activation('softmax')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model


################################
############# 学習 #############
################################
def learning(tsnum=10, nb_epoch=10, batch_size=8, learn_schedule=0.9):



    x_train, y_train, _ = load_images(TRAIN_DIR)
    x_valid, y_valid, _ = load_images(VALIDATE_DIR)

    print(">> 学習サンプル数 : ", np.array(x_train).shape)
    # 学習率の変更
    class Schedule(object):
        def __init__(self, init=0.001):  # 初期値定義
            self.init = init

        def __call__(self, epoch):  # 現在値計算
            lr = self.init
            for i in range(1, epoch + 1):
                lr *= learn_schedule
            return lr

    def get_schedule_func(init):
        return Schedule(init)

    lrs = LearningRateScheduler(get_schedule_func(0.001))
    mcp = ModelCheckpoint(filepath='best.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    hc = HistoryCheckpoint(filepath='LearningCurve_{history}.png'
                           , verbose=1
                           , period=2
                           )
    model = build_cnn()

    print("x_valid: ", np.shape(x_valid))
    print("y_valid: ", np.shape(y_valid))
    print("x_train: ", np.shape(x_train))
    print("y_train: ", np.shape(y_train))
    print(">> 学習開始")
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     verbose=1,
                     epochs=nb_epoch,
                     validation_data=(x_valid, y_valid),
                     callbacks=[lrs, mcp, hc])

    json_string = model.to_json()
    json_string += '##########' + str(CLASSES)
    open('model.json', 'w').write(json_string)
    model.save_weights('last.hdf5')


################################
########## 試行・実験 ##########
################################
def test_process():
    modelname_text = open("model.json").read()
    json_strings = modelname_text.split('##########')
    textlist = json_strings[1].replace("[", "").replace("]", "").replace("\'", "").split()
    model = model_from_json(json_strings[0])
    model.load_weights("last.hdf5")  # best.hdf5 で損失最小のパラメータを使用

    x_test,y_test,filenames = load_images(TRAIN_DIR)

    preds = model.predict(x_test, batch_size=1, verbose=0)


    collect = 0
    for i,pred in enumerate(preds):
        result = np.argmax(pred)
        if y_test[i][result] == 1:
            collect += 1
        print(filenames[i])
        print(str(pred))
        print(">> 「" + CLASSES[result] + "」")

    print(f"試験数: {len(y_test)}")
    print(f"正解数: {collect}")
    print(f"正解率: {collect / len(y_test)}")
    return result
