import os
import constants
import numpy as np
from PIL import Image

def resize(image, dim1, dim2):
    return image.resize((dim2, dim1), Image.LANCZOS)

def fileWalk(directory, destPath):
    try: 
        os.makedirs(destPath)
    except OSError:
        if not os.path.isdir(destPath):
            raise

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if len(file) <= 4 or file[-4:] != '.jpg':
                continue

            # 使用 PIL 打开图片
            pic = Image.open(os.path.join(subdir, file))
            # 转换为 numpy 数组以进行旋转操作
            pic_array = np.array(pic)
            dim1 = pic_array.shape[0]
            dim2 = pic_array.shape[1]
            
            if dim1 > dim2:
                pic_array = np.rot90(pic_array)
                pic = Image.fromarray(pic_array)

            # 调整图片大小并保存
            picResized = resize(pic, constants.DIM1, constants.DIM2)
            picResized.save(os.path.join(destPath, file), quality=95)

def main():
	prepath = os.path.join(os.getcwd(), 'dataset-original')
	glassDir = os.path.join(prepath, 'glass')
	paperDir = os.path.join(prepath, 'paper')
	cardboardDir = os.path.join(prepath, 'cardboard')
	plasticDir = os.path.join(prepath, 'plastic')
	metalDir = os.path.join(prepath, 'metal')
	trashDir = os.path.join(prepath, 'trash')

	destPath = os.path.join(os.getcwd(), 'dataset-resized')
	try: 
		os.makedirs(destPath)
	except OSError:
		if not os.path.isdir(destPath):
			raise

	#GLASS
	fileWalk(glassDir, os.path.join(destPath, 'glass'))

	#PAPER
	fileWalk(paperDir, os.path.join(destPath, 'paper'))

	#CARDBOARD
	fileWalk(cardboardDir, os.path.join(destPath, 'cardboard'))

	#PLASTIC
	fileWalk(plasticDir, os.path.join(destPath, 'plastic'))

	#METAL
	fileWalk(metalDir, os.path.join(destPath, 'metal'))

	#TRASH
	fileWalk(trashDir, os.path.join(destPath, 'trash'))  

if __name__ == '__main__':
    main()