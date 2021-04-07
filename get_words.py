import numpy as np
from scipy.cluster.vq import kmeans
import os
import cv2
import config

def get_all_desc_in_folder(path, detector, debug=False):
    if debug: print("READING FROM " + path)
    # i = 0
    list_desc = []
    list_dir = os.listdir(path)
    for child_path in list_dir:
        if debug: print("Getting descption from " + path + child_path)
        image = cv2.imread(path + child_path)
        image = config.pre_process_image(image)
        _, desc = detector.detectAndCompute(image, None) 
        # for index in desc.shape[1]:
        # elif not desc is None:
        if not desc is None:
            # list_desc = np.append(list_desc, desc, axis=0)
            list_desc.extend(desc.tolist())
        elif desc is None:
            print("ERROR: can\'t get the description from " + path + child_path)
            # list_desc = np.append(list_desc, np.zeros((1, list_desc.shape[1])), axis=0) 
            list_desc.extend([0] * len(list_desc[0]))
        # print(len(list_desc[0]))
        # i += 1
        # if i == 10:
        #     break
        # print(desc.shape)
        # print(list_desc.shape)
    return np.array(list_desc,).reshape(-1, len(list_desc[0])), list_dir
        
if __name__ == "__main__":
    list_desc = None
    list_dir = []
    for path in config.paths:
        desc, child_dir = get_all_desc_in_folder(path, config.detector, True)
        list_dir.extend(child_dir)
        if list_desc is None:
            list_desc = desc
        else:
            list_desc = np.append(list_desc, desc, axis=0)
    
    print(list_desc)
    np.savetxt("desc.txt", list_desc, fmt="%d")
    # print("CLUSTERING")
    # num_cluster = 1000
    # codebook, distortion = kmeans(list_desc, num_cluster, 1)
    # # print(codebook.shape)
    # np.savetxt("codebook.txt", codebook)
    # print("SAVING INDEXING")
    # with open("index.txt", "w") as file:
    #     print("\n".join(list_dir), file=file)
    # print("INITIALISE INVERT INDEX FILES")
    # for i in range(num_cluster):
    #     with open("./index/" + str(i) + ".txt", "w") as index_file:
    #         continue
