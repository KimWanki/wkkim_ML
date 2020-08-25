"""
authors : wkkim <seoulkwk95@gmail.com>
license : MIT
"""

import pydicom as dcm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
tree = ET.parse('/Users/kimwanki/Downloads/colon_cancer/00000001/1001097159_20100114/5_20100114/')
import matplotlib.pyplot as plt

import glob
from pydicom.data import get_testdata_file
from pydicom.filereader import read_dicomdir
import xml.etree.ElementTree as ET


path = '/Users/kimwanki/Downloads/colon_cancer/00000001/1001097159_20100114/5_20100114'
dcmlist = glob.glob(path+'/*')

dcm = [dcm.read_file(i) for i in dcmlist]
for i in dcm:
    print(i.filename)




