import h5py
import cv2
import os
import numpy as np
import pathlib
from random import shuffle
from random import seed

seed(111)


# n개의 수에서 2개를 뽑는 조합의 수를 계산하는 함수이다.
def combination_by_2(file_list):
    T_pair_list = []
    for i in range(len(file_list)):
        for n in range(i + 1, len(file_list)):
            T_pair_list.append((file_list[i], file_list[n]))
    return T_pair_list


train_data_directory = ('C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\05\\class1_2')
hdf5_path = r'C:\Users\kwjin\Desktop\handwriting\handwriting_image\06_hdf5\class1_2.hdf5'

train_paths = []    #siamese_data_set에 있는 파일경로 저장
test_paths = []
val_paths = []

# 페어 리스트 combination_by_2 하기전에 들어갈 리스트
T_train_list = []
T_val_list = []
T_test_list = []
train_num = 61
val_num = 29
test_num = 50


for i in range(1, train_num+1):
    T_train_list.append(str(i).zfill(3))
for i in range(1, val_num+1):
    T_val_list.append(str(i).zfill(3))
for i in range(1, test_num+1):
    T_test_list.append(str(i).zfill(3))

forder_num = 15
test_forder_num = 5


for (path, directory, files) in os.walk(train_data_directory + "/train"):
    df = 1
    for filename in files:
        ext = os.path.splitext(filename)[-1]

        if ext == '.jpg':
            train_paths.append(path + '/' + str(df).zfill(3) + ext)
        df += 1

# val_img 폴더 하위디렉토리에 있는 .jpg 파일을 다 읽어온다.

for (path, directory, files) in os.walk(train_data_directory + "/val"):
    df = 1
    for filename in files:
        ext = os.path.splitext(filename)[-1]

        if ext == '.jpg':
            val_paths.append(path + '/' + str(df).zfill(3) + ext)
        df += 1
# test_img 폴더 하위디렉토리에 있는 .jpg 파일을 다 읽어온다.
for (path, directory, files) in os.walk(train_data_directory + "/test"):
    df = 1
    for filename in files:
        ext = os.path.splitext(filename)[-1]

        if ext == '.jpg':
            test_paths.append(path + '/' + str(df).zfill(3) + ext)
        df += 1



        # (본인 img, 본인 img) pair를 만들기 위해 가능한 pair를 모두 계산한다.
        ## 1008개, 216개, 216개
T_train_pair_list = combination_by_2(T_train_list)

T_val_pair_list = combination_by_2(T_val_list)

T_test_pair_list = combination_by_2(T_test_list)


# train, validation, test paths에 존재하는 모든 실험자의 인덱스를 모은다.
train_prefix_list = []
val_prefix_list = []
test_prefix_list = []

for path in train_paths:
    prefix = pathlib.PurePath(path)
    prefix = prefix.parent.name
    # prefix = os.path.basename(os.path.normpath(path))
    if prefix not in train_prefix_list:
        train_prefix_list.append(prefix)

for path in val_paths:
    prefix = pathlib.PurePath(path)
    prefix = prefix.parent.name
    if prefix not in val_prefix_list:
        val_prefix_list.append(prefix)

for path in test_paths:
    prefix = pathlib.PurePath(path)
    prefix = prefix.parent.name
    if prefix not in test_prefix_list:
        test_prefix_list.append(prefix)


train_dir = train_data_directory + "/train"
val_dir = train_data_directory + "/val"
test_dir = train_data_directory + "/test"
# test set 만들때 :
# test_dir = "./kim_data_15_5_test_aug"

# 20장의 본인 이미지에서 (본인이미지 , 본인 이미지) 쌍이 가능한 경우의 수만큼 pair를 만들어주는 함수
# 20장 중에서 2개를 고르는 경우의 수는 190번의 경우가 생김, train_img에 포함된 70명 대상으로 13300 쌍이 만들어짐
def make_True_pairs(diretory, prefix_list, T_pair_list):
    T_pairs = []

    for prefix in prefix_list:
        for pairs in T_pair_list:
            head_path = diretory + '\\' + prefix + '/' + pairs[0] + '.jpg'
            tail_path = diretory + '\\' + prefix + '/' + pairs[1] + '.jpg'
            T_pairs.append((head_path, tail_path))

    return T_pairs


# (본인이미지 , 타인 이미지) pair를 만든다. (한명 당 200개의 쌍이 만들어진다.train_img에 포함된 70명 대상으로 14000 쌍이 만들어짐)
def Train_make_False_pairs(directory, prefix_list, total_paths):

    F_pairs = []

    for prefix in prefix_list:
        # shuffle(file_list)
        other_prefix_paths = []

        for path in total_paths:
            if prefix not in path:
                other_prefix_paths.append(path)

        shuffle(other_prefix_paths)

        for i in range(train_num-1):
            head_path = directory + '\\' + prefix + '/' + T_train_list[i] + '.jpg'
            for j in range(int(train_num/2)):
                tail_path = other_prefix_paths[j]
                F_pairs.append((head_path, tail_path))
    # csvfile = open("./running_data", "w", newline="")
    # csvwriter = csv.writer(csvfile)
    # for row in F_pairs:
    #     csvwriter.writerow(row)
    # csvfile.close()
    return F_pairs


def Val_make_False_pairs(directory, prefix_list, total_paths):
    file_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']

    F_pairs = []

    for prefix in prefix_list:
        # shuffle(file_list)
        other_prefix_paths = []

        for path in total_paths:
            if prefix not in path:
                other_prefix_paths.append(path)

        shuffle(other_prefix_paths)

        for i in range(val_num-1):
            head_path = directory + '\\' + prefix + '/' + T_val_list[i] + '.jpg'
            for j in range(int(val_num/2)):
                tail_path = other_prefix_paths[j]
                F_pairs.append((head_path, tail_path))

    return F_pairs


def Test_make_False_pairs(directory, prefix_list, total_paths):
    file_list = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']

    F_pairs = []

    for prefix in prefix_list:
        # shuffle(file_list)
        other_prefix_paths = []

        for path in total_paths:
            if prefix not in path:
                other_prefix_paths.append(path)

        shuffle(other_prefix_paths)

        for i in range(test_num-1):
            head_path = directory + '\\' + prefix + '/' + T_test_list[i] + '.jpg'
            for j in range(int(test_num/2)):
                tail_path = other_prefix_paths[j]
                F_pairs.append((head_path, tail_path))

    return F_pairs

# 없 단어 T_pairs
T_train_path_pairs_01 = make_True_pairs(train_dir, train_prefix_list[0:forder_num], T_train_pair_list)
T_val_path_pairs_01 = make_True_pairs(val_dir, val_prefix_list[0:forder_num], T_val_pair_list)
T_test_path_pairs_01 = make_True_pairs(test_dir, test_prefix_list[0:test_forder_num], T_test_pair_list)
# # 다 단어 T_pairs
# T_train_path_pairs_02 = make_True_pairs(train_dir, train_prefix_list[10:], T_train_pair_list)
# T_val_path_pairs_02 = make_True_pairs(val_dir, val_prefix_list[20:], T_val_pair_list)
# T_test_path_pairs_02 = make_True_pairs(test_dir, test_prefix_list[20:], T_test_pair_list)
# 없, 다 최종 T_paris
T_train_path_paris_final = T_train_path_pairs_01 #+ T_train_path_pairs_02
T_val_path_paris_final = T_val_path_pairs_01 #+ T_val_path_pairs_02
T_test_path_paris_final = T_test_path_pairs_01 #+ T_test_path_pairs_02

# 없 단어 F_pairs
F_train_path_pairs_01 = Train_make_False_pairs(train_dir, train_prefix_list[0:forder_num], train_paths)
F_val_path_pairs_01 = Val_make_False_pairs(val_dir, val_prefix_list[0:forder_num], val_paths)
F_test_path_pairs_01 = Test_make_False_pairs(test_dir, test_prefix_list[0:test_forder_num], test_paths)
# 다 단어 F_pairs
# F_train_path_pairs_02 = Train_make_False_pairs(train_dir, train_prefix_list[20:], train_paths[1120:])
# F_val_path_pairs_02 = Val_make_False_pairs(val_dir, val_prefix_list[20:], val_paths[240:])
# F_test_path_pairs_02 = Test_make_False_pairs(test_dir, test_prefix_list[20:], test_paths[240:])
# 없, 다 최종 F_paris
F_train_path_paris_final = F_train_path_pairs_01 #+ F_train_path_pairs_02
F_val_path_paris_final = F_val_path_pairs_01 #+ F_val_path_pairs_02
F_test_path_paris_final = F_test_path_pairs_01 #+ F_test_path_pairs_02

# T_pairs label
T_train_labels = [1] * len(T_train_path_paris_final)
T_val_labels = [1] * len(T_val_path_paris_final)
T_test_labels = [1] * len(T_test_path_paris_final)
# F_pairs label
F_train_labels = [0] * len(F_train_path_paris_final)
F_val_labels = [0] * len(F_val_path_paris_final)
F_test_labels = [0] * len(F_test_path_paris_final)

######################################################################
# #대용량파일 만들기위한 공간확보
T_train_set = list(zip(T_train_path_paris_final, T_train_labels))
F_train_set = list(zip(F_train_path_paris_final, F_train_labels))

T_val_set = list(zip(T_val_path_paris_final, T_val_labels))
F_val_set = list(zip(F_val_path_paris_final, F_val_labels))

T_test_set = list(zip(T_test_path_paris_final, T_test_labels))
F_test_set = list(zip(F_test_path_paris_final, F_test_labels))

train_set = T_train_set + F_train_set
val_set = T_val_set + F_val_set
test_set = T_test_set + F_test_set

# train, validation, test img들의 path가 담긴 list 들을 섞는다.
shuffle(train_set)
shuffle(val_set)
shuffle(test_set)

train_paths, train_labels = zip(*train_set)
val_paths, val_labels = zip(*val_set)
test_paths, test_labels = zip(*test_set)

trainset_len = len(train_labels)
val_set_len = len(val_labels)
test_set_len = len(test_labels)

train_shape = (trainset_len, 2, 112, 112, 3)
val_shape = (val_set_len, 2, 112, 112, 3)
test_shape = (test_set_len, 2, 112, 112, 3)

############### hdf5 만들기 ##################################


hdf5_file = h5py.File(hdf5_path, mode='w')

# 27300장(train) 및 5850장(test&validation set)의 이미지 쌍을 저장하기 위한 empty array 생성
hdf5_file.create_dataset("train_img", train_shape, np.uint8)
hdf5_file.create_dataset("val_img", val_shape, np.uint8)
hdf5_file.create_dataset("test_img", test_shape, np.uint8)

# label 은 empty array 생성 후 바로 hdf5 파일에 저장
hdf5_file.create_dataset("train_labels", (trainset_len,), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (val_set_len,), np.int8)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (test_set_len,), np.int8)
hdf5_file["test_labels"][...] = test_labels

# # train 이미지 hdf5 파일에 저장
for i in range(trainset_len):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, trainset_len))
    # cv2 load images as BGR, convert it to RGB
    l_addr = train_paths[i][0]
    r_addr = train_paths[i][1]
    l_img = cv2.imread(l_addr)
    r_img = cv2.imread(r_addr)
    l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)

    # save the image
    hdf5_file["train_img"][i, 0, ...] = l_img[None][None]  # np.shape(1,1,112,112,3)
    hdf5_file["train_img"][i, 1, ...] = r_img[None][None]  # np.shape(1,1,112,112,3)


# validation 이미지 hdf5 파일에 저장
for i in range(val_set_len):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Validation data: {}/{}'.format(i, val_set_len))
    # cv2 load images as BGR, convert it to RGB
    l_addr = val_paths[i][0]
    r_addr = val_paths[i][1]
    l_img = cv2.imread(l_addr)
    r_img = cv2.imread(r_addr)
    l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    # save the image
    hdf5_file["val_img"][i, 0, ...] = l_img[None][None]  # np.shape(1,1,112,112,3)
    hdf5_file["val_img"][i, 1, ...] = r_img[None][None]  # np.shape(1,1,112,112,3)

# test 이미지 hdf5 파일에 저장
for i in range(test_set_len):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, test_set_len))
    # cv2 load images as BGR, convert it to RGB
    l_addr = test_paths[i][0]
    r_addr = test_paths[i][1]
    l_img = cv2.imread(l_addr)
    r_img = cv2.imread(r_addr)
    l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2RGB)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    # save the image
    hdf5_file["test_img"][i, 0, ...] = l_img[None][None]  # np.shape(1,1,112,112,3)
    hdf5_file["test_img"][i, 1, ...] = r_img[None][None]  # np.shape(1,1,112,112,3)

hdf5_file.close()



# # hdf5파일 잘 되었는지 확인
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
#
# hdf5_path = './hdf5_training/class2_4.hdf5'
#
# # open the hdf5 file
# hdf5_file = h5py.File(hdf5_path, "r")
#
# # Total number of samples
# data = hdf5_file["test_img"][100:110,...]
# label = hdf5_file["test_labels"][100:110,...]
#
# print(data.shape[2])
# print(data.shape[3])
# print(np.shape(data))
# print(type(data))
#
# for i in range(10):
#     fig, ax = plt.subplots(1, 2, figsize=(10, 10))
#     ax[0].set_axis_off()
#     ax[0].set_title(label[i], size=30)
#     ax[0].imshow(data[i][0])
#     ax[1].set_axis_off()
#     ax[1].imshow(data[i][1])
#     plt.show()
#     plt.close()