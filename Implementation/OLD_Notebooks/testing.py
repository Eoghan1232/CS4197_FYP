import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

# Path for Large Movie Dataset
path_M = r'G:/FYP_Work/FYP_Datasets/Large_Movie_dataset/aclImdb/train/pos//'
positive_M_files = glob.glob(path_M + "*.txt")
li = []


for filename in positive_M_files:
    temp = np.loadtxt(filename,dtype=str,encoding="UTF-8")
    li.append(temp)
li = [temp for temp in li if not temp.size == 0]

print(li[10])
# Path for Reuters Dataset
path_R = r'FYP_Datasets\\'