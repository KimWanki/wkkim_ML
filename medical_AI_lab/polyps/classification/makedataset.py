import shutil, os, cv2, numpy as np
import matplotlib.pyplot as plt
list = ['Adeno','HGD','LGD','Normal']

# input img,mask path
def mkRec(img):
    image = cv2.imread(img)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()
    h, w = image.shape[0], image.shape[1]

    # TODO : input size를 정사각형 형태로 맞춰주기.
    import math
    # # TODO : 가로로 붙이기.
    if h > w :
        image_bg = np.zeros(shape=(h, math.ceil((h - w) / 2), 3), dtype=np.uint8)
        image = cv2.hconcat([image_bg, image, image_bg])
    # TODO : 세로로 붙이기.
    elif h < w :
        image_bg = np.zeros(shape=(math.ceil((w-h)/2), w, 3), dtype=np.uint8)
        image = cv2.vconcat([image_bg, image, image_bg])

    return image


base_path = 'C:/Users/USER/Desktop/wkkim/classification/train/'

Tcopy_path = 'C:/Users/USER/Desktop/wkkim/classification/dataset/train/'
Vcopy_path = 'C:/Users/USER/Desktop/wkkim/classification/dataset/val/'
TScopy_path = 'C:/Users/USER/Desktop/wkkim/classification/dataset/test/'

img_path = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/train'
val_path = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/val'
test_path = 'C:/Users/USER/Desktop/wkkim/segmentation/TestDataset/test'
_train_data = os.listdir(img_path)
_val_data = os.listdir(val_path)
_test_data = os.listdir(test_path)


for i in _train_data:
    for j in os.listdir(base_path+list[0]):
        if i == j:
            cv2.imwrite(Tcopy_path+list[0]+'/'+i,mkRec(base_path+list[0]+'/'+i))
            # shutil.copy(base_path+list[0]+'/'+i,Tcopy_path+list[0]+'/'+i)
            break
    for j in os.listdir(base_path+list[1]):
        if i == j:
            cv2.imwrite(Tcopy_path + list[1] + '/' + i, mkRec(base_path + list[1] + '/' + i))
            # shutil.copy(base_path+list[1]+'/'+i,Tcopy_path+list[1]+'/'+i)
    for j in os.listdir(base_path+list[2]):
        if i == j:
            cv2.imwrite(Tcopy_path + list[2] + '/' + i, mkRec(base_path + list[2] + '/' + i))
            # shutil.copy(base_path+list[2]+'/'+i,Tcopy_path+list[2]+'/'+i)
    for j in os.listdir(base_path+list[3]):
        if i == j:
            cv2.imwrite(Tcopy_path + list[3] + '/' + i, mkRec(base_path + list[3] + '/' + i))
            # shutil.copy(base_path+list[3]+'/'+i,Tcopy_path+list[3]+'/'+i)

for i in _val_data:
    for j in os.listdir(base_path+list[0]):
        if i == j:
            cv2.imwrite(Vcopy_path + list[0] + '/' + i, mkRec(base_path + list[0] + '/' + i))
            # shutil.copy(base_path+list[0]+'/'+i,Vcopy_path+list[0]+'/'+i)
            break
    for j in os.listdir(base_path+list[1]):
        if i == j:
            cv2.imwrite(Vcopy_path + list[1] + '/' + i, mkRec(base_path + list[1] + '/' + i))
    for j in os.listdir(base_path+list[2]):
        if i == j:
            cv2.imwrite(Vcopy_path + list[2] + '/' + i, mkRec(base_path + list[2] + '/' + i))
    for j in os.listdir(base_path+list[3]):
        if i == j:
            cv2.imwrite(Vcopy_path + list[3] + '/' + i, mkRec(base_path + list[3] + '/' + i))

for i in _test_data:
    for j in os.listdir(base_path+list[0]):
        if i == j:
            cv2.imwrite(TScopy_path + list[0] + '/' + i, mkRec(base_path + list[0] + '/' + i))
            break
    for j in os.listdir(base_path+list[1]):
        if i == j:
            cv2.imwrite(TScopy_path + list[1] + '/' + i, mkRec(base_path + list[1] + '/' + i))
    for j in os.listdir(base_path+list[2]):
        if i == j:
            cv2.imwrite(TScopy_path + list[2] + '/' + i, mkRec(base_path + list[2] + '/' + i))
    for j in os.listdir(base_path+list[3]):
        if i == j:
            cv2.imwrite(TScopy_path + list[3] + '/' + i, mkRec(base_path + list[3] + '/' + i))



