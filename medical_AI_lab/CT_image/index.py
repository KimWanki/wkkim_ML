"""
authors : wkkim <seoulkwk95@gmail.com>
license : MIT
"""
import pydicom as dcm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# tree = ET.parse('/Users/kimwanki/Downloads/colon_cancer/00000001/1001097159_20100114/5_20100114/')

import os
import glob
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree

def makexml(filename, width, height, point):
    label = 'polyps'
    root = Element('annotation')
    SubElement(root, 'folder').text = 'custom_images'
    SubElement(root, 'filename').text = filename + '.png'
    SubElement(root, 'path').text = './object_detection/images' + filename + '.png'
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '1'

    SubElement(root, 'segmented').text = '0'

    obj = SubElement(root, 'object')
    SubElement(obj, 'name').text = label
    SubElement(obj, 'pose').text = 'Unspecified'
    SubElement(obj, 'truncated').text = '0'
    SubElement(obj, 'difficult').text = '0'
    bbox = SubElement(obj, 'bndbox')
    SubElement(bbox, 'xmin').text = str(point[1])
    SubElement(bbox, 'ymin').text = str(point[2])
    SubElement(bbox, 'xmax').text = str(point[3])
    SubElement(bbox, 'ymax').text = str(point[4])

    tree = ElementTree(root)
    tree.write('./' + filename + '.xml')


folder_path = '/Users/kimwanki/developer/colon_cancer/'
folder_list = os.listdir(folder_path)
folder_list.remove('.DS_Store')
print(folder_list)
path = folder_path + folder_list[2]
oslist= os.listdir(path)
dcmlist = glob.glob(path+'/*')
dcmlist = glob.glob(path+'/[!default]*')

dcmlist = sorted(dcmlist)
print(len(dcmlist))

xml = glob.glob(path+'/**/*.xml')
xml = ET.parse(xml[0])
root = xml.getroot()

_dcm = [dcm.read_file(i) for i in dcmlist]

location_list =[]
for Contour in root.findall('Contour'):
    slice = Contour.find('Slice-number')
    pt = [i.text for i in Contour.findall('Pt')]
    print(pt)
    _min, _max = pt[0].replace('\'','').replace('.0',''), pt[2].replace('\'','').replace('.0','')
    _xmin, _ymin = _min.split(',')
    _xmax, _ymax = _max.split(',')
    print(_min, " " , _max)

    if int(_xmin) > int(_xmax):
        _xmin,_xmax = _xmax, _xmin
    if float(_ymin) > float(_ymax):
        _ymin,_ymax = _ymax, _ymin

    location = [int(slice.text), int(_xmin), int(_ymin), int(_xmax), int(_ymax)]
    location_list.append(location)
location_list = sorted(location_list)
for i in location_list:
    width, height = _dcm[(len(dcmlist) - 1) - i[0]].pixel_array.shape

    plt.imshow(_dcm[(len(dcmlist)-1)-i[0]].pixel_array)

    matplotlib.image.imsave(folder_list[2]+'_'+str(i[0])+'.png', _dcm[(len(dcmlist)-1)-i[0]].pixel_array)
    makexml(folder_list[2]+'_'+str(i[0]),width,height,i)

    ax = plt.gca()
    print( "i : ",i)
    print("dx:", i[3]-i[1], " dy:", i[4]-i[2])
    rect = patches.Rectangle((i[1], i[2]),
                             i[3]-i[1],
                             i[4]-i[2],
                             linewidth=1,
                             edgecolor='cyan',
                             fill=False)
    ax.add_patch(rect)
    plt.show()
