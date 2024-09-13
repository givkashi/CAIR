import os
from PIL import Image
import numpy as np
dir1 = r"G:\codes\lama-main\celeba-hq-dataset\visual_test_source_256"
dir2 = r"G:\codes\lama-main\celeba-hq-dataset\visual_test_256\random_thick_256"
a=0
np.zeros_like
for filename in os.listdir(dir1):
    if os.path.isfile(os.path.join(dir2, filename[:-4] + "_crop000.png")):
        a += 1
    else:
        print(filename)
        image = Image.open(os.path.join(dir1, filename))
        mask = Image.fromarray(np.zeros_like(np.array(image)))
        image.save(os.path.join(dir2, filename[:-4] + "_crop000.png"))
        mask.save(os.path.join(dir2, filename[:-4] + "_crop000_mask000.png"))

print(a)