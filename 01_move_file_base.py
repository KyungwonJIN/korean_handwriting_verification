from random import randrange, choice
from math import ceil
from pathlib import Path
import shutil
import os

start_num_1st = int(input('move file start num : '))

dirpath = Path('C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\00_tight_crop/p')
movepath = Path('C:\\Users\\kwjin\\Desktop\\handwriting\\handwriting_image\\01_move_image/01')

last_folder_name = 1
directory_num = int(input('dir_num : '))

directory = []
for i in range(1, directory_num + 1):
    directory.append('p' + str(i).zfill(4))

for pdir in directory:
    (movepath / pdir).mkdir(exist_ok=True)

# 1~20까지의 범위로 os.walk를 통해 가져온 path, directory, files로 파일을 옮긴다.
for i in range(1, 21):
    last_name = str(i).zfill(4)
    for (path, directory, files) in os.walk(str(dirpath) + last_name):

        new_name = 1
        val_name = 1
        test_name = 1

        start_num = start_num_1st

        for filename in files:
            ext = os.path.splitext(filename)[-1]  ## 확장자


            new_path = str(new_name).zfill(3) + ext
            new_name += 1

            train_paths = path + '\\' + str(start_num).zfill(3) + ext
            start_num += 1

            last_folder_path = "p" + str(last_name).zfill(4)

            shutil.copy(str(train_paths), str(movepath / str(last_folder_path) / new_path))

            if new_name == 11:
                break
            else:
                print(movepath / str(last_folder_path) / new_path)
        last_folder_name += 1
