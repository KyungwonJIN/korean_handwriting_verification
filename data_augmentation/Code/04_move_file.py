from random import randrange, choice
from math import ceil
from pathlib import Path
import shutil
import os
from tqdm import tqdm

print("train,val,test 갯수 입력")
train_num = int(input('train_num : '))
val_num = int(input('val_num : '))
test_num = int(input('test_num : '))
directory_num = int(input('dir_num : '))

# train_paths = [] #siamese_data_set에 있는 파일경로 저장
test_paths = []
val_paths = []
directory = []



for d in tqdm(range(1, 5)):
    for i in range(1, directory_num + 1):
        directory.append('p' + str(i).zfill(4))
    dirpath = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\num_140\\' + str(d).zfill(2) + '\p')
    movepath = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\04_segmentation'
                    r'\\class' + str(d) + '')
    for subdir in ['train', 'val', 'test']:
        (movepath / subdir).mkdir(exist_ok=True)
        for pdir in directory:
            (movepath / subdir / pdir).mkdir(exist_ok=True)

    for i in range(1, directory_num + 1):
        last_name = str(i).zfill(4)
        for (path, directory, files) in os.walk(str(dirpath) + last_name):
            new_name = 0
            val_name = 1
            test_name = 1
            num = 1
            for filename in files:
                ext = os.path.splitext(filename)[-1]    ## 확장자명  ex) .jpg
                tt = os.path.splitext(filename)[0]      ## 확장자명 제외한 경로. ex) c:\\users\picture\001 (.jpg 제외)
                # kkk = os.path.splitext(path)
                last_folder_name = os.path.split(path)  ## 경로와 파일을 분리. ex) 'c:\\users~~\\', 'p0001' <-파일명 분리됨.
                # print(last_folder_name + ' last')
                tt = int(tt)

                if num <= train_num:
                    new_name = new_name
                    new_name += 1
                elif num > train_num and num <= train_num + val_num:
                    new_name = val_name
                    val_name += 1
                else:
                    new_name = test_name
                    test_name += 1
                num += 1
                #
                # if new_name < train_num :
                #     new_name = new_name
                #     new_name += 1
                # elif new_name < train_num + val_num :
                #     new_name = val_name
                #     val_name += 1
                # else :
                #     new_name = test_name
                #     test_name += 1
                new_path = str(new_name).zfill(3) + ext
                # print("dd", type(tt), tt, last_folder_name,last_folder_name[1])
                #
                # if ext == '.jpg':
                train_paths = path + '/' + str(tt).zfill(3) + ext
                # df += 1
                # # fn = int(path.split('\\')[2][0:3])
                # kkk = os.path.splitext(path)
                # kkk = os.path.split(kkk[0])
                # print("확장자제외" +kkk[1])
                # fn = int(path.splitext([0]))
                # print(fn)
                # fn = int(str(train_paths.relative_to(dirpath).with_suffix('')))
                subdir = 'train' if tt <= train_num else ('val' if tt <= train_num + val_num else 'test')
                shutil.copy(str(train_paths), str(movepath / subdir / last_folder_name[1] / new_path))
