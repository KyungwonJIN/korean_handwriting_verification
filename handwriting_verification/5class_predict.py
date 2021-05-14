import cv2
import numpy as np
from keras.models import load_model
import h5py
from keras import backend as K
import csv
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import timeit

CUDA_VISIBLE_DEVICES=0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

K.clear_session()

file_name = 'class5_1'
weight_name = 'class5_309-0.0421_modify'

# PRED_COODS_FILE = "siamese_model/parrot_siamese.csv" #예측한 좌표값 저장파일이름
PRED_COODS_FILE = './5class_predict/' + file_name + '_mod' + '.csv'
PRED_IMAGE_FILE = './5class_predict/'

# hdf5_path = './dataset/resnet_kim_10_sgd_oldmodel16-0.0028.hdf5' #입력데이터

#입력데이터

# hdf5_path = './hdf5_genuine/class1/'+ file_name + '.hdf5'
hdf5_path = './5class_hdf5/' + file_name + '.hdf5'
# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")
# validation_data = hdf5_file["val_img"][...]
# validation_label = hdf5_file["val_labels"][...]
train_data = hdf5_file["test_img"][...]
train_label = hdf5_file["test_labels"][...]
# print(train_data.shape)
hdf5_file.close()
print('data_load_state : done')

pred_coods = open(PRED_COODS_FILE, "w", newline='')
writer = csv.writer(pred_coods, delimiter=",")

# model = load_model('./handwriting_resnet02-0.5324.hdf5') #저장된 모델
model = load_model('./5class_modify_weight/' + weight_name + '.hdf5')  #저장된 모델
model.summary()
start_time = timeit.default_timer()
from keras.models import Model
train_model = Model(model.input, model.get_layer('dense_2').output)



predicted_pre = train_model.predict([train_data[:, 0], train_data[:, 1]])
# print([train_data[:,0]])
# print(type(train_data))
predicted_binary = (predicted_pre > 0.5).astype(np.int)
# print(predicted_binary)

# preds = train_model.predict([train_data[:, 0], train_data[:, 1]])
# for cls in training_generator.class_indices:
#     print(cls+": "+preds[0][training_generator.class_indices[cls]])

# generate_grad_cam(train_data, model, train_label, dense_2)


# predicted_pre = model.predict([validation_data[:, 0], validation_data[:, 1]])
real_label = train_label
# score = model.evaluate([train_data[:, 0], train_data[:, 1]], train_label, verbose=1)
# print('training_end\n')
# print('Test loss: {:.4f}'.format(score[0]))
# print('Test accuracy: {:.4f}'.format(score[1]))
# print('\nAccuracy: {:.4f}'.format(model.evaluate([train_data[:, 0], train_data[:, 1]], train_label, verbose=0)[1]))


# real_label = validation_label

print(len(train_data))
count = 0
far = 0
frr = 0
for i in range(len(train_data)):
    pre_value = float(predicted_pre[i])
    pre_label = int(predicted_binary[i])
    save_label = real_label[i]
    row = [pre_value, pre_label, save_label]
    if(predicted_binary[i] > real_label[i]):
        far +=1
    elif(predicted_binary[i] < real_label[i]):
        frr +=1
    else:  #predicted_binary[i] == real_label[i]):
        count +=1
    ## 저장 빼고 잠깐 test. 2021/04/28
    ## sigmoid층에서 자동으로 분류 되는가
    # writer.writerow(row)
print("accuracy = ", count/(len(train_data)))
# print(count)
# print("FAR = ", far, ", FRR = ", frr)
# print("testing_end")
pred_coods.close()

terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time))




# pre_label_output = np.zeros(len(train_data), dtype=int)
# for a in range(len(train_data)):
#     pre_label_output[a] = int(predicted_binary[a])
#
# wrong_result = []
#
# for n in range(0, len(train_label)):
#     if pre_label_output[n] == train_label[n]:
#         wrong_result.append(n)
#
# samples = random.choices(population=wrong_result, k=5)
#
# for i in samples:
#     fig, ax = plt.subplots(1, 2)
#     tmp = "Label: " + str(train_label[i]) + ", Prediction: " + str(pre_label_output[i])
#     # tmp2 = "Prediction: " + str(pre_label_output[i])
#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
#     # ax[0].set_axis_off()
#     ax[0].set_title(tmp, loc='right')
#     ax[0].imshow(train_data[i][0])
#     # ax[1].set_axis_off()
#     ax[1].set_xticks([])
#     ax[1].set_yticks([])
#     # ax[1].set_title(tmp2, loc='left')
#     ax[1].imshow(train_data[i][1])
#     # plt.show()
#     fig.savefig(PRED_IMAGE_FILE + file_name+ '_' + str(i) + '.jpg')
#     plt.close()



# for i in samples:
#     #print("label:", train_labels[i])
#     plt.imshow(train_data[i][0],train_data[i][1])
#     #temp = "Label:" + str(train_label[i]) + ", i : " + str(i)
#     #plt.title(temp)
#     tmp = "Label: " + str(train_label[i]) + " , Prediction: " + str(pre_label_output[i])
#     plt.title(tmp)
#     fig = plt.gcf()
#     plt.show()
#     fig.savefig(PRED_IMAGE_FILE+str(i)+'.jpg')


