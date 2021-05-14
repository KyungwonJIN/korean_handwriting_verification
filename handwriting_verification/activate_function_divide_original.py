import resnet_woohyuk_utill as resnet
from keras import backend as K
from keras.layers import Input, Activation
from keras.layers.core import Lambda, Dense
from keras.models import Model
from keras.regularizers import l2


data_shape = (112, 112, 3)
img_rows, img_cols = 112, 112
img_channels = 3
feature_vecs = 4096
class_name = 'class3_4_03-0.1185'

siamese_network =resnet.get_resnet((img_channels, img_rows, img_cols), feature_vecs)
siamese_network.summary()

left_input = Input(shape=data_shape)
right_input = Input(shape=data_shape)
L_model = siamese_network(left_input)
R_model = siamese_network(right_input)

L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([L_model, R_model])
logit = Dense(1, kernel_initializer='he_normal', kernel_regularizer=l2(2e-4))(L1_distance)
prediction = Activation('sigmoid')(logit)
model = Model(inputs=[left_input, right_input], outputs=prediction)
model.load_weights('E:\handwriting_kw\필적_training_model/' + class_name + '.hdf5')
model.summary()

model.save('./modify_weight/' + class_name + '_modify_test.hdf5')