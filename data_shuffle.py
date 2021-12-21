import os
import glob
import shutil

src_dir = 'dataset\ddown'
dst_dir = 'dataset\data_shuffled'
for png in glob.iglob(os.path.join(src_dir, "*.png")):
    shutil.copy(png, dst_dir)