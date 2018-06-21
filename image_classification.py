import pandas as pd
import numpy as np
import os
#import imageio

from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Maximum
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize as imresize
from skimage import io
import matplotlib.image as mpimg
from tqdm import tqdm

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train_data = pd.read_csv('train.csv')
train_data_dropped = train_data.drop_duplicates(['Id'])
train_data_dropped = train_data_dropped.reset_index(drop = True)
id_series = train_data_dropped['Id']
class_label = pd.DataFrame({'class': id_series.index, 'label':id_series.values})
CLASS_NUM = len(class_label)
CLASS = class_label.set_index('label').T.to_dict(orient = 'records')[0]
INV_CLASS = class_label.set_index('class').T.to_dict(orient = 'records')[0]
for i in range(len(train_data)):
    label = train_data.loc[i, 'Id']
    train_data.loc[i, 'class'] = CLASS[label]


# Dense layers set
def dense_set(inp_layer, n, activation, drop_rate=0.):
    dp = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act


# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1, 1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    if activation == 'leakyRL':
        act = LeakyReLU(1 / 10)(bn)
    elif activation == 'sigmoid':
        act = Activation(activation='sigmoid')(bn)
    elif activation == 'tanh':
        act = Activation(activation='tanh')(bn)
    return act


# simple model
def get_model(opt='sgd'):
    inp_img = Input(shape=(51, 51, 3))

    # 51
    conv1 = conv_layer(inp_img, 64, activation='leakyRL', zp_flag=False)
    conv2 = conv_layer(conv1, 64, zp_flag=False)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
    # 23
    conv3 = conv_layer(mp1, 128, zp_flag=False)
    conv4 = conv_layer(conv3, 128, zp_flag=False)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    # 9
    '''''
    conv7 = conv_layer(mp2, 256, zp_flag=False)
    conv8 = conv_layer(conv7, 256, zp_flag=False)
    conv9 = conv_layer(conv8, 256, zp_flag=False)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)
    '''
    # 1
    # dense layers
    flt = Flatten()(mp2)
    ds1 = dense_set(flt, 128, activation='sigmoid')
    out = dense_set(ds1, CLASS_NUM, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)

    # The first 50 epochs are used by Adam opt.
    # Then 30 epochs are used by SGD opt.
    if opt == 'adam':
        mypotim = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif opt == 'sgd':
        mypotim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    model.summary()
    return model


def get_callbacks(filepath, patience=5):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1)
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [lr_reduce, msave]


# I trained model about 12h on GTX 950.
def train_model(X_train, y_train, opt, BATCH_SIZE = 16, EPOCHS = 30, RANDOM_STATE = 11):
    if opt == 'adam':
        callbacks = get_callbacks(filepath='model_weight_Adam.hdf5', patience=6)
        gmodel = get_model(opt)
    elif opt == 'sgd':
        callbacks = get_callbacks(filepath='model_weight_SGD.hdf5', patience=6)
        gmodel = get_model(opt)
        gmodel.load_weights(filepath='model_weight_Adam.hdf5')
    x_train, x_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        shuffle=True,
        train_size=0.8,
        random_state=RANDOM_STATE
    )
    gen = ImageDataGenerator(
        rotation_range=360.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )
    gmodel.fit_generator(gen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                         steps_per_epoch=10 * len(x_train) / BATCH_SIZE,
                         epochs=EPOCHS,
                         verbose=1,
                         shuffle=True,
                         validation_data=(x_valid, y_valid),
                         callbacks=callbacks)


def test_model(X_test, img_name):
    gmodel = get_model()
    gmodel.load_weights(filepath='model_weight_SGD.hdf5')
    prob = gmodel.predict(X_test, verbose=1)
    pred = []
    for p in prob:
        prob_df = pd.DataFrame({'Prob': p})
        top_prob = prob_df.sort_values(by='Prob', ascending=False)[0:5]
        sum = 0
        for i in range(5):
            sum += top_prob.iloc[i]['Prob']
        percent_prob = top_prob
        for i in range(5):
            percent_prob.iloc[i]['Prob'] = top_prob.iloc[i]['Prob'] / sum
        index = percent_prob.index
        pred.append(index)
    operation = 'all'
    if operation == 'all':
        id_list = []
        for p in pred:
            str = ''
            for i in range(5):
                str = str + ' ' + INV_CLASS[p[i]]
            id_list.append(str)
        #id_list = [[INV_CLASS[p[i]] for i in range(5)] for p in pred]
    elif operation == 'one':
        pred = prob.argmax(axis=-1)
    sub = pd.DataFrame({"Image": img_name,
                        "Id": id_list})
    sub.to_csv("submission.csv", index=False, header=True)


# Resize all image to 51x51
def img_reshape(img):
    img = imresize(img, (51, 51, 3))
    return img


# get image tag
def img_label(path):
    image_name = str(str(path.split('/')[-1]))
    label = train_data[(train_data.Image==image_name)]['Id'].values[0]
    return label

# get plant class on image
def img_class(path):
    image_name = str(str(path.split('/')[-1]))
    class_num = train_data[(train_data.Image == image_name)]['class'].values[0]
    return class_num


# fill train and test dict
def fill_dict(paths, some_dict):
    text = ''
    if 'train' in paths[0]:
        text = 'Start fill train_dict'
    elif 'test' in paths[0]:
        text = 'Start fill test_dict'

    for p in tqdm(paths, ascii=True, ncols=85, desc=text):
        #img = imageio.imread(p)
        img = io.imread(p)
        img = img_reshape(img)
        some_dict['image'].append(img)
        if 'train' in paths[0]:
            some_dict['label'].append(img_label(p))
            some_dict['class'].append(img_class(p))
        elif 'test' in paths[0]:
            image_name = str(str(p.split('/')[-1]))
            some_dict['name'].append(image_name)

    return some_dict


# read image from dir. and fill train and test dict
def reader():
    file_ext = []
    train_path = []
    test_path = []

    for root, dirs, files in os.walk('/input'):  #'input' for computer, '/input' for floydhub
        if dirs != []:
            print('Root:\n' + str(root))
            print('Dirs:\n' + str(dirs))
        else:
            for f in files:
                ext = os.path.splitext(str(f))[1][1:]

                if ext not in file_ext:
                    file_ext.append(ext)

                if 'train' in root:
                    path = os.path.join(root, f)
                    train_path.append(path)
                elif 'test' in root:
                    path = os.path.join(root, f)
                    test_path.append(path)
    train_dict = {
        'image': [],
        'label': [],
        'class': []
    }
    test_dict = {
        'image': [],
        'name': []
    }

    train_dict = fill_dict(train_path, train_dict)
    test_dict = fill_dict(test_path, test_dict)
    return train_dict, test_dict


# I commented out some of the code for learning the model.
def main():
    train_dict, test_dict = reader()
    operation = 'test'
    if operation == 'train':
        X_train = np.array(train_dict['image'])
        y_train = to_categorical(np.array([CLASS[l] for l in train_dict['label']]))
        train_model(X_train, y_train, 'adam', EPOCHS=50)
        train_model(X_train, y_train, 'sgd', EPOCHS=30)
    elif operation == 'test':
        X_test = np.array(test_dict['image'])
        img_name = test_dict['name']
        test_model(X_test, img_name)
    elif operation == 'both':
        X_train = np.array(train_dict['image'])
        y_train = to_categorical(np.array([CLASS[l] for l in train_dict['label']]))
        train_model(X_train, y_train, 'adam', EPOCHS=50)
        train_model(X_train, y_train, 'sgd', EPOCHS=30)
        X_test = np.array(test_dict['image'])
        img_name = test_dict['name']
        test_model(X_test, img_name)



if __name__ == '__main__':
    main()