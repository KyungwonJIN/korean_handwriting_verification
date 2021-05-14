# hdf5파일 잘 되었는지 확인
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg
from keras.models import load_model
import tensorflow as tf

import timeit

CUDA_VISIBLE_DEVICES=0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config = config)




hdf5_path = 'E:\handwriting_kw\필적_trainingdata_nfolded/class4_4.hdf5'
# open the hdf5 file
hdf5_file = h5py.File(hdf5_path, "r")

# Total number of samples
## 50:100 --> hdf5파일 내에서 원하는 부분을 불러올 수 있음.
data = hdf5_file["test_img"][50:100, ...]
label = hdf5_file["test_labels"][50:100, ...]


answer = []
corr = 0
wrong = 0
print(type(corr))

weight_name = 'class4_4_15-0.0159_modify'

model = load_model('./mm_modify_weight/' + weight_name + '.hdf5')
start_time = timeit.default_timer()
from keras.models import Model

test_model = Model(model.input, model.get_layer('dense_2').output)
predicted_pre = test_model.predict([data[:, 0], data[:, 1]])



for i in range (len(predicted_pre)) :
    if predicted_pre[i] > -2:
        answer.append(1)
    else:
        answer.append(0)

predict_sum = 0

for i in range(len(predicted_pre)):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set_axis_off()
    fname0 = label[i]
    fname1 = answer[i]
    fname1_1 = predicted_pre[i]
    ax[0].set_title('Real_label : {}'
                    '\nPredict_label : {} '
                    '\nPredict_value : {}'.format(fname0, fname1, ("%.2f"% predicted_pre[i])), size=30)
    ax[0].imshow(data[i][0])

    predict_sum += predicted_pre[i]

    # 예측 값
    ax[1].set_xlabel(predicted_pre[i])

    ax[1].set_axis_off()
    ax[1].imshow(data[i][1])
    # plt.show()


    print("predict_val : ", predicted_pre[i], "answer :" , answer[i], "label_val : ", label[i])

    ## 이부분에서 정답과 오답을 색을 나눠서 저장한다.
    ## 저장할 파일 위치 savefig('') 에 입력해주면된다.
    if label[i] == answer[i]:
        corr += 1
        fig.savefig('E:\handwriting_kw\com for predict2/' + str(i).zfill(3) + '.jpg',
                    bbox_inches='tight')
    else:
        wrong += 1
        fig.savefig('E:\handwriting_kw\com for predict2/' + str(i).zfill(3) + '.jpg',
                    facecolor='xkcd:salmon', bbox_inches='tight')
        plt.show()
    plt.close()
print(predict_sum)
from PIL import Image, ImageDraw, ImageFont
import os, glob

print('correct num : ' + str(corr) + '\n' + 'wrong num : ' + str(wrong) + '\n' + 'accuracy : ' + str(corr/(corr+wrong)))
terminate_time = timeit.default_timer()
print("%f초 걸렸습니다." % (terminate_time - start_time))
