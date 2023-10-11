from PIL import Image
import os

dir_list = os.listdir()
i = 1

for filename in dir_list:
    if ".png" in filename:
        with Image.open(filename) as img:
            img.load()
            FTB_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            FLR_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            R90_img = img.transpose(Image.ROTATE_90)
            R180_img = img.transpose(Image.ROTATE_180)
            FTB_img.save(str(i+100) + ".png")
            FLR_img.save(str(i+200) + ".png")
            R90_img.save(str(i+300) + ".png")
            R180_img.save(str(i+400) + ".png")
            i += 1

