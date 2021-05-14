
# hdf5파일 잘 되었는지 확인
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg

## 없김다이 4글자 한번에 test 하기위해서 4번 돌림
for j in range(4):
    hdf5_path = 'E:\handwriting_kw\필적_trainingdata_nfolded/class'+ str(j+1) +'_1.hdf5'
    # E:\필적_trainingdata_nfolded
    # open the hdf5 file
    hdf5_file = h5py.File(hdf5_path, "r")

    ## 1000:1050 번까지 50개의 사진을 띄워주며 테스트 진행
    # Total number of samples
    data = hdf5_file["test_img"][1000:1050, ...]
    label = hdf5_file["test_labels"][1000:1050, ...]

    answer = []
    corr = 0
    wrong = 0
    print(type(corr))

    for i in range(50):
        print(i)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].set_axis_off()
        # ax[0].set_title(label[i], size=30)
        ax[0].imshow(data[i][0])

        ax[1].set_axis_off()
        ax[1].imshow(data[i][1])
        plt.show()

        while True:
            try :
                k = int(input("same = 1, not same = 0 : "))
            except (ValueError):
                print('please enter integer')
                continue
            if k == 1 or k == 0 :
                answer.append(k)
                break
            else :
                print('1 or 0을 입력하시오')
                continue
        ax[1].set_title(answer[i], position=(0.5, 0.5), size=50, fontname='times new roman', y=0.5)

        ## 예측한 사진들 저장하는 폴더
        if label[i] == answer[i]:
            corr += 1
            fig.savefig('E:\handwriting_kw\필적_trainingdata_nfolded/image_result_'+ str(j+1) +'/' + str(i).zfill(3) + '.jpg', bbox_inches='tight')
        else:
            wrong +=1
            fig.savefig('E:\handwriting_kw\필적_trainingdata_nfolded/image_result_'+ str(j+1) +'/' + str(i).zfill(3) + '.jpg',
                        facecolor='xkcd:salmon', bbox_inches='tight')
            # plt.show()
        plt.close()

    from PIL import Image, ImageDraw, ImageFont
    import os, glob

    for i in range(50):
        target_image = Image.open('E:\필적_trainingdata_nfolded/image_result_'+ str(j+1) +'/'+ str(i).zfill(3) + '.jpg')
        fontsFolder = r'C:\Users\kwjin\Anaconda3\envs\keras2.2.4-gpu\Lib\site-packages\matplotlib\mpl-data\fonts\ttf'
        selectedFont = ImageFont.truetype(os.path.join(fontsFolder, 'timesnewroman.ttf'),50)
        draw = ImageDraw.Draw(target_image)
        draw.text((40,10), 'ANSWER : ' + str(label[i]), fill='black', font=selectedFont)
        target_image.save('E:\필적_trainingdata_nfolded/image_result_answer_'+ str(j+1) +'/' + str(i).zfill(3) + '.jpg')

    print('correct num : ' + str(corr) + '\n' + 'wrong num : ' + str(wrong) + '\n' + 'accuracy : ' + str(corr/(corr+wrong)))