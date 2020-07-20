import cv2
import matplotlib.pyplot as plt


img_path = '/Users/kimwanki/developer/ML/imgFile/9.jpeg'
xml_path = '/Users/kimwanki/developer/ML/imgFile/9.xml'

a = cv2.imread(img_path)
a = cv2.rotate(a , cv2.ROTATE_90_COUNTERCLOCKWISE)
print(a.shape)
# plt.imshow(a)
# plt.show()

location_list = []

with open(xml_path,'r') as xml_read:
    # print(xml_read.readlines())
    for i in xml_read.readlines():
        if 'ymin' in i:
            x =  i.replace('\t','').replace('\n','').replace('<ymin>','').replace('</ymin>','')
            location_list.append(int(x))
        elif 'xmin' in i:
            x =  i.replace('\t','').replace('\n','').replace('<xmin>','').replace('</xmin>','')
            location_list.append(int(x))
        elif 'ymax' in i:
            x = i.replace('\t', '').replace('\n', '').replace('<ymax>', '').replace('</ymax>', '')
            location_list.append(int(x))
        elif 'xmax' in i:
            x = i.replace('\t', '').replace('\n', '').replace('<xmax>', '').replace('</xmax>', '')
            location_list.append(int(x))

a = a[location_list[1]:location_list[3],location_list[0]:location_list[2],:]
plt.imshow(a)
plt.show()

# lotation =

print(location_list)






