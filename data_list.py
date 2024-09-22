import pandas as pd 
import glob
import re
import os

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
os.chdir('dataset')
files = sorted(glob.glob('train/*.png'), key=natural_keys)
print(files)
with open('./train.txt', 'w') as f:
    for d in files:
  
        f.write("%s\n" % d)
