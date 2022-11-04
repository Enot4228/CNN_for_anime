from skimage import io
import numpy as np

image = io.imread('./data/Yofukashi no Uta/yofukashi_no_uta_1.jpg')
print(np.mean(image / 255.))
#print(np.mean(image / 255.))

