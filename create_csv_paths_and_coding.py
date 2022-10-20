import pandas as pd
import os
from pathlib import Path


root_dir = Path('data').parent
data_dir = root_dir / 'data'
list_class_code = []
list_path = []
list_class_decode = []
list_class_encode = []
counter = 0
for i in os.listdir(data_dir):
    class_dir = data_dir / i
    print(class_dir)
    for j in os.listdir(class_dir):
        image_path = class_dir / j
        list_class_code.append(counter)
        list_path.append(image_path)
    list_class_decode.append(i)
    list_class_encode.append(counter)
    counter += 1

df = pd.DataFrame(list_path, columns=['Path'])
df['class'] = list_class_code
df_code = pd.DataFrame(list_class_decode, columns=['class'])
df_code['class_code'] = list_class_encode
df.to_csv('./path_dataset.csv')
df_code.to_csv('./class_codes.csv')
