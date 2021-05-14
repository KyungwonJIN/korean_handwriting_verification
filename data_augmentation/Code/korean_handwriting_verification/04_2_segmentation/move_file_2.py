from random import randrange, choice
from math import ceil
from pathlib import Path
import shutil
import os
from tqdm import tqdm
import operator

original_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\04_segmentation\\class')
move_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\05\\class')
test_folder = [1, 2, 3, 4, 5]
five = [5, 5, 5, 5, 5]
for i in range(1, 5):
    for q in range(1, 5):
        for k in range(0, 20):
            original_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\'
                                 r'04_segmentation\\class' + str(i) + r'\\test\p' + str(k + 1).zfill(4))
            move_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\05\\class'
                             r'' + str(i) + '_' + str(q) + r'\\test\p' + str(k + 1).zfill(4))
            train_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\'
                              r'04_segmentation\\class' + str(i) + r'\\train\p' + str(k + 1).zfill(4))
            movet_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\05\\class'
                              r'' + str(i) + '_' + str(q) + r'\\train\p' + str(k - 4).zfill(4))
            val_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\'
                            r'04_segmentation\\class' + str(i) + r'\\val\p' + str(k + 1).zfill(4))
            movev_path = Path(r'C:\\Users\kwjin\Desktop\handwriting\handwriting_image\\05\\class'
                              r'' + str(i) + '_' + str(q) + r'\\val\p' + str(k - 4).zfill(4))
            print(test_folder[0], k + 1, test_folder[4])

            if test_folder[0] <= k + 1 <= test_folder[4]:
                # subdir_test = '' + str(i) + r'\\test\p' + str(k + 1).zfill(4)
                # mvdir_test = '' + str(i) + '_' + str(q) + r'\\test'

                shutil.copytree(str(original_path),
                                str(move_path))
            else:
                shutil.copytree(str(train_path),
                                str(movet_path))
                shutil.copytree(str(val_path),
                                str(movev_path))
        map(operator.add, test_folder, five)
    test_folder = [1, 2, 3, 4, 5]
