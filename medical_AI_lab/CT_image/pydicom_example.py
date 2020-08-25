import pydicom as dcm


import matplotlib.pyplot as plt
import glob
import glob
from pydicom.data import get_testdata_file
from pydicom.filereader import read_dicomdir

path = './0002.DCM'
img = dcm.dcmread(path)



for i in range(len(img.pixel_array[:,0,0])):
    plt.imshow(img.pixel_array[i], cmap=plt.cm.gray)
    plt.show()