import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import resnet_team as resnet
import resnet_woohyuk_utill as resnet1
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers import Input, Activation
from keras.layers.core import Lambda, Dense
from keras.optimizers import SGD, RMSprop
import tensorflow as tf

from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import multi_gpu_model

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc=0)

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc=0)

hdf5_path = './hdf5_training/class1_1.hdf5'
file_name = 'class1_1'

# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")

train_data = hdf5_file["train_img"][...]
train_label = hdf5_file["train_labels"][...]
validation_data = hdf5_file["val_img"][...]
validation_label = hdf5_file["val_labels"][...]
test_data = hdf5_file["test_img"][...]
test_label = hdf5_file["test_labels"][...]
hdf5_file.close()
print('data_load_state : done')
data_shape = np.shape(train_data[1, 0])

MODEL_SAVE_FOLDER_PATH = './model/resnet34/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

# model_path = MODEL_SAVE_FOLDER_PATH + 'siamese_parrot_resnet' + '{epoch:02d}-{val_loss:.4f}.hdf5'

model_path = MODEL_SAVE_FOLDER_PATH + file_name + '{epoch:02d}-{val_loss:.4f}.hdf5'

# lr_reducer는 콜백함수로 model.fit 함수가 동작하면서 학습이 침체되는 경우가 생기면 learning late 를 줄인다.
# ReduceLROnPlateau 함수는 검승 손실이 향상되지 않을 떄 학습률(learning late)를 줄인다.
# 줄이는게 맞는지 키우는게 맞는지?
cb_lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, cooldown=0, patience=10, min_lr=0.5e-6, verbose=1, mode='min')
# epoch 결과를 .csv 파일에 스트리밍해준다.
# cv_csv_logger = CSVLogger('siamese_parrot_resnet.csv')
cv_csv_logger = CSVLogger(file_name + '.csv')
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
# EarlyStopping 함수는 에폭을 주고 그 에폭보다 향상되지 않은면 중단시킨다.
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=7, mode='min')



# input image dimensions
ROW_AXIS, COL_AXIS = 112, 112
img_rows, img_cols = 112, 112
img_channels = 3
CHANNEL_AXIS = 3
#????
feature_vecs = 4096


siamese_network =resnet1.get_resnet((img_channels, img_rows, img_cols), feature_vecs)
# siamese_network.summary()

# siamese_network = resnet.ResNet18(input_shape=( ROW_AXIS, COL_AXIS,  CHANNEL_AXIS), classes = 1, block='basic', residual_unit='v1',
#            repetitions=None, initial_filters=64, activation='sigmoid', include_top=True,
#            input_tensor=None, dropout=0.5, transition_dilation_rate=(1, 1),
#            initial_strides=(1, 1), initial_kernel_size=(7, 7), initial_pooling='max',
#            final_pooling='avg', top='classification')



# siamese_network =resnet.ResNet((img_channels, img_rows, img_cols), feature_vecs)
# siamese_network =resnet1.get_denseNet((CHANNEL_AXIS, ROW_AXIS, COL_AXIS), feature_vecs)
from keras.preprocessing import image
siamese_network.summary()
tf.keras.utils.plot_model(
    siamese_network, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB'
)
# grad_img = image.load_img('./001.jpg', target_size=(112, 112))
# grad_img = image.img_to_array(grad_img)
# generate_grad_cam(grad_img, siamese_network, 0, 'activation_17')

left_input = Input(shape=data_shape)
right_input = Input(shape=data_shape)
L_model = siamese_network(left_input)
R_model = siamese_network(right_input)



import cv2
# l_img1 = cv2.imread('./single_test/001.jpg')
# r_img1 = cv2.imread('./single_test/002.jpg')
# l_img1 = cv2.cvtColor(l_img1, cv2.COLOR_BGR2RGB)
# r_img1 = cv2.cvtColor(r_img1, cv2.COLOR_BGR2RGB)
#
# l_img1 = np.expand_dims(l_img1, axis=0)
# r_img1 = np.expand_dims(r_img1, axis=0)




L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([L_model, R_model])
prediction = Dense(1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(2e-4))(L1_distance)
# logit = Dense(1, kernel_initializer='he_normal', kernel_regularizer=l2(2e-4))(L1_distance)
# prediction = Activation('sigmoid')(logit)



model = Model(inputs=[left_input, right_input], outputs=prediction)
# model = multi_gpu_model(model, gpus=2)

sgd = SGD(lr=0.1, decay=1e-3, momentum=0.99, nesterov=True) # lr= 1e-2 or 1e-3

optimizer = Adam(lr=0.001)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
print('history')







history = model.fit([train_data[:, 0], train_data[:, 1]], train_label,
                    validation_data=([validation_data[:, 0], validation_data[:, 1]], validation_label), epochs=200,
                    batch_size=8, verbose=1, callbacks=[cb_checkpoint, cb_early_stopping, cb_lr_reducer, cv_csv_logger])

score = model.evaluate([test_data[:, 0], test_data[:, 1]], test_label, verbose=1)

print('training_end\n')
print('Test loss: {:.4f}'.format(score[0]))
print('Test accuracy: {:.4f}'.format(score[1]))
print('\nAccuracy: {:.4f}'.format(model.evaluate([test_data[:, 0], test_data[:, 1]], test_label, verbose=0)[1]))
print(history)

plot_loss(history)
plt.show()
plot_acc(history)
plt.show()

# Accuracy
# print(history)
# fig1, ax_acc = plt.subplots()
# plt.plot(history.history['acc'])'
# plt.plot(history.history['val_acc'])
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Model - Accuracy')
# plt.legend(['Training', 'Validation'], loc='lower right')
# plt.show()




