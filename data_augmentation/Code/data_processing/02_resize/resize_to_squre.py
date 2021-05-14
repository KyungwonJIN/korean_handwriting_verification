import numpy as np
import cv2
import matplotlib as plt

for k in range(1, 21):

    for j in range(1, 11):  # image 갯수

        """경로 및 파일 이름 설정 부분
        """
        path_last = str(k).zfill(2)

        image_second = str(j).zfill(3)

        imageSource = r'C:\Users\kwjin\Desktop\handwriting\handwriting_image\01_move_image\04\p00' + path_last + "\\" + image_second + ".jpg"
        # imageSource = "single_test/02100001.jpg"
        # imageSave = image_first + image_second

        img = cv2.imread(imageSource)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        height, width = img.shape[:2]
        # height_bigger = int(height-width)/2
        # width_bigger = int(width-heigth)/2

        if height > width:
            square_img = cv2.copyMakeBorder(img, 0, 0, int((height-width)/2), int((height-width)/2), cv2.BORDER_CONSTANT, value=(255, 255, 255))
        elif width < height:
            square_img = cv2.copyMakeBorder(img, int((width-height)/2), int((width-height)/2), 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else :
            square_img = img

        # resize_img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        # print(height , ",", width)
        resize_img = cv2.resize(square_img, dsize=(48, 48), interpolation=cv2.INTER_AREA)


        # cv2.imshow("letter", resize_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        final_image = cv2.copyMakeBorder(resize_img, 32, 32, 32, 32,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
        savename = "C:\\Users\kwjin\Desktop\handwriting\\handwriting_image\\02_resize_image\\04\\p00" + path_last + "/" + str(j).zfill(3) + ".jpg"
        # savename = "single_test/02100001_1.jpg"
        cv2.imwrite(savename, final_image)
        cv2.waitKey(0)
# from PIL import Image, ImageEnhance
# path = "C:\\Users\kwjin\Desktop\handwriting\\handwriting_tight_crop\\test2\p0001/000.jpg"
# path2 = "C:\\Users\kwjin\Desktop\handwriting\\handwriting_tight_crop\\test2\p0001/0000.jpg"

# reimg = Image.open("C:\\Users\kwjin\Desktop\handwriting\\handwriting_tight_crop\\test2\p0001/001.jpg")
# # dpi = reimg.info['dpi']
# # reimg.info['dpi']
# # dpi = (80, 80)
# simg = ImageEnhance.Sharpness(reimg)
# reimg.save(path, dpi=(200,200))
# # simg.save(path2, dpi=(200,200))
# simg.enhance(-2.0).show()
