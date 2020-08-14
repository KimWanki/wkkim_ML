"""
authors : wkkim <seoulkwk95@gmail.com>
license : MIT
"""

import pydicom as dcm
import matplotlib.pyplot as plt
import glob
import glob
from pydicom.data import get_testdata_file
from pydicom.filereader import read_dicomdir

path = 'C:/Users/user/Downloads/colon_cancer/colon_cancer/00000001/1001097159_20100114/5_20100114'
dcmlist = glob.glob('C:/Users/user/Documents/5_20100114/*')
img = dcm.dcmread(dcmlist)
plt.imshow(img)
plt.show()


