import numpy as np
from scipy.cluster.vq import kmeans
import os
import cv2
import config


def get_all_desc_in_folder(path, detector, debug=False):
    if debug: print("READING FROM " + path)
    list_desc = None
    for child_path in os.listdir(path):
        if debug: print("Getting descption from " + path + child_path)
        image = cv2.imread(path + child_path)
        image = config.pre_process_image(image)
        _, desc = detector.detectAndCompute(image, None) 
        # for index in desc.shape[1]:
        if list_desc is None:
            list_desc = desc
        else:
            list_desc = np.append(list_desc, desc, axis=0)

        # print(desc.shape)
        # print(list_desc.shape)
    return list_desc
        
if __name__ == "__main__":
    list_desc = None
    for path in config.paths:
        desc = get_all_desc_in_folder(path, config.detector, True)
        if list_desc is None:
            list_desc = desc
        else:
            list_desc = np.append(list_desc, desc, axis=0)

    codebook, distortion = kmeans(list_desc, 200, 1)
    # print(codebook.shape)
    np.savetxt("codebook.txt", codebook)
