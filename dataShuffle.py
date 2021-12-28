import os
import glob
import shutil

src_dir = 'dataset\\dnothing'
dst_dir = 'dataset\\shuffled\\'
for jpg in glob.iglob(os.path.join(src_dir, "*.jpg")):
    finalPath = dst_dir + jpg[17:int(len(jpg))-4] + "_" + str(src_dir[9:len(src_dir)]) + ".jpg"
    shutil.copy(jpg, finalPath)