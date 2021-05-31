import zipfile
import csv

from secml.array import CArray
from secml.data import CDataset
from secml.utils import fm

# Path to a folder with 'feature_vectors.zip' and 'sha256_family.csv' files
# Available at: https://www.sec.tu-bs.de/~danarp/drebin/download.html
DREBIN_DIR = ''
DREBIN_FEAT_VECT_ZIP = 'feature_vectors.zip'
DREBIN_MALS = 'sha256_family.csv'

DREBIN_FEAT_VECT_PATH = fm.join(DREBIN_DIR, DREBIN_FEAT_VECT_ZIP)
DREBIN_MALS_PATH = fm.join(DREBIN_DIR, DREBIN_MALS)

# Dictionary of malware samples (app hash is the key)
mals_dict = {}
with open(DREBIN_MALS_PATH, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)
    for row in reader:
        mals_dict[row[0]] = row[1]

unzipped_file = zipfile.ZipFile(DREBIN_FEAT_VECT_PATH, "r")

# List of files inside 'feature_vectors.zip'
files = []
for item in unzipped_file.infolist():
    if item.is_dir():
        continue
    files.append(item.filename)

# Map each feature to an unique ID and create the dict of samples and labels
# Feature name is the hash of 'feats_dict', app hash is the key
# for 'samples_x_dict' and 'samples_y_dict'
feats_dict = {}
samples_x_dict = {}
samples_y_dict = {}
for s_path in sorted(files):
    s_name = fm.split(s_path)[-1]  # App hash from filename
    with unzipped_file.open(s_path) as sf:
        s = sf.readlines()
        feats = []  # List of feature IDs
        for feat in s:
            feat = feat.decode("utf-8")
            feat = feat.replace('\n', '')
            if feat not in feats_dict:
                feats_dict[feat] = len(feats_dict)
            feats.append(feats_dict[feat])
        samples_x_dict[s_name] = feats
    # If app hash is in the 'mals_dict' label should be 1
    samples_y_dict[s_name] = 1 if s_name in mals_dict else 0

# Create the arrays of samples and labels
x = CArray.zeros(shape=(len(samples_x_dict), len(feats_dict)),
                 dtype=int, sparse=True)
y = CArray.zeros(shape=(len(samples_x_dict),), dtype=int)

x = x.tolil()  # lil is more efficient to build the array

for s_i, s in enumerate(samples_x_dict):
    if s_i % 1000 == 0:
        print(s_i)
    x[s_i, samples_x_dict[s]] = 1
    y[s_i] = samples_y_dict[s]

x = CArray(x.tocsr())

# Create and save the final dataset
ds = CDataset(x, y)
ds.save(fm.join(DREBIN_DIR, 'drebin.tar.gz'))
