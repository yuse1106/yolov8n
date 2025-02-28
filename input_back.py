# 入力画像に異なる背景画像にする
import cv2
import numpy as np
import os

number = 20
for i in range(number):
    image_path = 'datasets_new_back/images/'
    os.makedirs(image_path+'train/', exist_ok=True)
    os.makedirs(image_path+'val/', exist_ok=True)
    # 画像の読み込み
    if i < 16:
        image = cv2.imread(f'datasets_new_back/images/train/image{i}.png')
    else:
        image = cv2.imread(f'datasets_new_back/images/val/image{i}.png')
    back_img = cv2.imread('../Testpython/image/back_30.jpg')
    back_img = cv2.resize(back_img, (416,416))

    # 白色の部分をマスクして
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thre, mask = cv2.threshold(gray_img, 20, 40, cv2.THRESH_BINARY)

    # マスク画像の反転
    mask_inv = cv2.bitwise_not(mask)

    # 画像と背景を合成
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    back_portion = cv2.bitwise_and(back_img, back_img, mask=mask)
    final_img = cv2.add(result, back_portion)

    # cv2.imshow('output', final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if i < 16:
        cv2.imwrite(image_path+f'train/image{i+16}.png', final_img)
    else:
        cv2.imwrite(image_path+f'val/image{i+16}.png', final_img)
