# import the OpenCV package
import cv2
import csv
import glob
import os
import numpy as np
from tqdm import tqdm

# example of random rotation image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from random import *
from pathlib import Path

directory = []
for i in range(1, 21):
    directory.append('p' + str(i).zfill(4))

for q in range(1, 5):
    movepath = Path(r'C:\Users\kwjin\Desktop\handwriting\handwriting_image\num_140\\'
                      + str(q).zfill(2))

    for pdir in directory:
        (movepath / pdir).mkdir(exist_ok=True)

for num in tqdm(range(1, 5)):


    folder_name = r'C:\Users\kwjin\Desktop\handwriting\handwriting_image\02_resize_image\\'\
                  + str(num).zfill(2) + '\p00'

    savefolder_name = r'C:\Users\kwjin\Desktop\handwriting\handwriting_image\test\\'\
                      + str(num).zfill(2) + '\p00'

    forder_num = 21 # +1해서 입력


    repeat = int(input('만들고싶은 총 갯수 : '))//10
    print('repeat : ', repeat)

    import random

    list = []
    ran_num = random.randint(1, repeat*10)
    for y in range(repeat*10):
        while ran_num in list:
            ran_num = random.randint(1, repeat*10)
        list.append(ran_num)
    print(list)








    def image_zoom():
        for k in tqdm(range(1, forder_num)):



            for j in range(1, 11):  # image 갯수

                """경로 및 파일 이름 설정 부분
                """
                path_last = str(k).zfill(2)

                # image_first = '01_'
                image_second = str(j).zfill(3)
                imageSource = folder_name + path_last + "/" + image_second + ".jpg"

                """경로 및 파일 이름 설정 부분
                """

                img = cv2.imread(imageSource)
                # convert to numpy array
                data = img_to_array(img)
                # expand dimension to one sample//차원을 늘리는 함수
                samples = expand_dims(data, 0)
                # create image data augmentation generator
                datagen = ImageDataGenerator(zoom_range=0.1,
                                             height_shift_range=0.03,
                                             width_shift_range=0.03,
                                             rotation_range=5, fill_mode='constant', cval=255)
                # prepare iterator
                it = datagen.flow(samples, batch_size=1)
                # generate samples and plot
                for i in range(0, repeat):
                    # define subplot
                    # pyplot.subplot(330 + 1 + i)
                    # generate batch of images
                    batch = it.next()
                    # convert to unsigned integers for viewing
                    image = batch[0].astype('uint8')
                    # plot raw pixel data
                    # pyplot.imshow(image)
                    ## range(3)일때
                    print('ori: ',str(j).zfill(3), 'new:',  str(list[(j-1)*repeat+i]).zfill(3))
                    savename = savefolder_name + path_last + "/" + str(list[(j-1)*repeat+i]).zfill(3) + ".jpg"
                    # print((j-1)*9+i+1)
                    # print(savename)
                    # savename = savefolder_name + path_last + "/" + str(30+j).zfill(3) + ".jpg"
                    resize_img = cv2.resize(image, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(savename, resize_img)



                # show the figure


    # 메인
    def main():



        # rotation 4회, scaling 4회, translation 4회
        # flip = image_flip()
        # shear = image_shear()
        ## rotation -> transmition -> scaling
        # rotate = rotation()
        # translation = image_translation()
        zoom = image_zoom()



    if __name__ == "__main__":
        main()
























