import cv2
import numpy as np
from pathlib import Path
import os

root_dir = Path('./video/Yofukashi no Uta')
counter = 1
for i in os.listdir(root_dir):
    print(root_dir / i)
    cap = cv2.VideoCapture(str(root_dir / str(i)))
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frames = cap.read()
        norm_frames = frames / 255.
        if np.mean(norm_frames) > 0.02 and counter % 12 == 0:
            cv2.imwrite('./data/Yofukashi no Uta/yofukashi_no_uta_' + str(counter) + '.jpg', frames)
            counter += 1
        else:
            counter += 1
            continue

        if counter % 1000 == 0:
            print(counter)


    cap.release()
    cv2.destroyAllWindows()