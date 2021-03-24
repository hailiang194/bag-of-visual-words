from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.cluster.vq import vq
import numpy as np
import os
import config
import cv2

def get_all_hist_in_folder(path, detector, codebook, debug=False):
    if debug: print("READING FROM " + path)
    list_hist = None
    child = []
    for child_path in os.listdir(path):
        child.append(path + child_path)
        if debug: print("Getting descption from " + path + child_path)
        image = cv2.imread(path + child_path)
        image = config.pre_process_image(image)
        _, desc = detector.detectAndCompute(image, None) 
        words, _ = vq(desc, codebook)
        hist = np.zeros((1, codebook.shape[0]))
        if debug: print("Getting it word histogram")
        for word in words.tolist():
            hist[0, word] += 1
        
        hist = hist / np.linalg.norm(hist)

        if list_hist is None:
            list_hist = hist
        else:
            list_hist = np.append(list_hist, hist, axis=0)
    return list_hist, child
 
if __name__ == "__main__":
    codebook = np.loadtxt("./codebook.txt")
    print("GENERATING DATASET")
    
    y = None
    label = {}
    X = None
    child = []
    for index, path in enumerate(config.paths):
        list_hist, child_paths = get_all_hist_in_folder(path, config.detector, codebook, True)  
        child.extend(child_paths)
        label[path] = index
        
        if X is None:
            X = list_hist
        else:
            X = np.append(X, list_hist, axis=0)
    
    np.savetxt("all_hist.txt", X) 
    # with open("image_path.txt", "w") as imageFile:
    #     for c in child:
    #         imageFile.write(c + "\n")
    # print("LEARNING")
    # clf = LinearSVC(max_iter=1000000)
    # clf.fit(X, y)
    # np.savetxt("weight.txt", clf.coef_)
    # with open("labels.txt", "w") as label_file:
    #     for key in label.keys():
    #         label_file.write(str(key) + " " + str(label[key]) + "\n")
