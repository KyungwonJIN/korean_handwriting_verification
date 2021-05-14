import cv2
import numpy as np
from keras.models import load_model
import h5py
from keras import backend as K
import csv
import matplotlib.pyplot as plt
import random
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

CUDA_VISIBLE_DEVICES=0

K.clear_session()
weight_name = 'class5_309-0.0421_modify'
# model = load_model('./handwriting_resnet02-0.5324.hdf5') #저장된 모델
# model = load_model('./modify_weight/class4_4_01-0.5056_modify.hdf5')  # 저장된 모델
model = load_model('./5class_modify_weight/' + weight_name + '.hdf5')
model.summary()
from keras.models import Model

test_model = Model(model.input, model.get_layer('dense_2').output)

# for i in range(0, 3):
file_name = 'single_modify'

# PRED_COODS_FILE = "siamese_model/parrot_siamese.csv" #예측한 좌표값 저장파일이름
PRED_COODS_FILE = './5class_single/' + file_name + '.csv'
PRED_IMAGE_FILE = './5class_single/'

pred_coods = open(PRED_COODS_FILE, "w", newline='')
writer = csv.writer(pred_coods, delimiter=",")

## 아래 이미지 두개에 원하는 사진이름 두개 넣어주면 됨.
# ing_num1 = str(1).zfill(3)
# ing_num2 = str(2).zfill(3)

# l_img = cv2.imread('./single_test/' + ing_num1 + '.jpg')
# r_img = cv2.imread('./single_test/' + ing_num2 + '.jpg')

ing_num1 = str(8).zfill(3)
ing_num2 = str(8).zfill(3)
l_img = cv2.imread('./single_test/' + ing_num1 + '.jpg')
r_img = cv2.imread('./single_test/' + ing_num2 + '_re.jpg')
l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)

l_img = np.expand_dims(l_img, axis=0)
r_img = np.expand_dims(r_img, axis=0)

predicted_pre = test_model.predict([l_img, r_img])
# print([test_data[:,0]])
# print(type(test_data))
predicted_binary = (predicted_pre > -2).astype(np.int)

print("predict 값", predicted_pre)
    # preds = test_model.predict([test_data[:, 0], test_data[:, 1]])
    # for cls in training_generator.class_indices:
    #     print(cls+": "+preds[0][training_generator.class_indices[cls]])

    # generate_grad_cam([l_img, r_img], model, 1,  model.get_layer('dense_2'))


    # predicted_pre = model.predict([validation_data[:, 0], validation_data[:, 1]])
    # real_label = 1
    # score = test_model.evaluate([[l_img], [r_img]], 1, verbose=1)
    # score = model.evaluate([test_data[:, 0], test_data[:, 1]], test_label, verbose=1)
    # print('training_end\n')
    # print('Test loss: {:.4f}'.format(score[0]))
    # print('Test accuracy: {:.4f}'.format(score[1]))
    # print('\nAccuracy: {:.4f}'.format(model.evaluate([[l_img], [r_img]], 1, verbose=1)))


# real_label = validation_label

# print(len(test_data))
# count = 0
# far = 0
# frr = 0
# for i in range(len(test_data)):
#     pre_value = float(predicted_pre[i])
#     pre_label = int(predicted_binary[i])
#     save_label = real_label[i]
#     row = [pre_value, pre_label, save_label]
#     if(predicted_binary[i] > real_label[i]):
#         far +=1
#     elif(predicted_binary[i] < real_label[i]):
#         frr +=1
#     else:#  predicted_binary[i] == real_label[i]):
#         count +=1
#     writer.writerow(row)
# print(count)
# print("FAR = ", far, ", FRR = ", frr)
# print("testing_end")
pred_coods.close()


#
# pre_label_output = np.zeros(len(test_data), dtype=int)
# for a in range(len(test_data)):
#     pre_label_output[a] = int(predicted_binary[a])
#
# wrong_result = []
#
# for n in range(0, len(test_label)):
#     if pre_label_output[n] == test_label[n]:
#         wrong_result.append(n)
#
# samples = random.choices(population=wrong_result, k=5)
#
# for i in samples:
#     fig, ax = plt.subplots(1, 2)
#     tmp = "Label: " + str(test_label[i]) + ", Prediction: " + str(pre_label_output[i])
#     # tmp2 = "Prediction: " + str(pre_label_output[i])
#     ax[0].set_xticks([])
#     ax[0].set_yticks([])
#     # ax[0].set_axis_off()
#     ax[0].set_title(tmp, loc='right')
#     ax[0].imshow(test_data[i][0])
#     # ax[1].set_axis_off()
#     ax[1].set_xticks([])
#     ax[1].set_yticks([])
#     # ax[1].set_title(tmp2, loc='left')
#     ax[1].imshow(test_data[i][1])
#     # plt.show()
#     fig.savefig(PRED_IMAGE_FILE + str(i) + '.jpg')
#     plt.close()
#
#
#
# for i in samples:
#     #print("label:", test_labels[i])
#     plt.imshow(test_data[i])
#     #temp = "Label:" + str(test_label[i]) + ", i : " + str(i)
#     #plt.title(temp)
#     tmp = "Label: " + str(test_label[i]) + " , Prediction: " + str(pre_label_output[i])
#     plt.title(tmp)
#     fig = plt.gcf()
#     # plt.show()
#     fig.savefig(PRED_IMAGE_FILE+str(i)+'.jpg')
#

