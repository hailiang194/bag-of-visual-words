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
    # i = 0
    image_path = None
    with open("./index.txt", "w") as list_image_file:
        image_path = os.listdir(path)
        print("\n".join(image_path), file=list_image_file)
        for index, child_path in enumerate(image_path):
            if debug: print("Getting descption from %r" % (path + child_path))
            image = cv2.imread(path + child_path.strip())
            image = config.pre_process_image(image)
            _, desc = detector.detectAndCompute(image, None) 
            words, _ = vq(desc, codebook)
            hist = np.zeros((1, codebook.shape[0]))
            if debug: print("Getting it word histogram")
            for word in words.tolist():
                hist[0, word] += 1
        
            hist = hist / np.sum(hist)
         
            #for i in range(codebook.shape[0]):
                #if hist[0, i] > 0:
                    #with open("./index/" + str(i) + ".txt", "a") as index_file:
                    #    print(str(index), file=index_file)
            if list_hist is None:
                list_hist = hist
            else:
                list_hist = np.append(list_hist, hist, axis=0)
            # i += 1
            # if i == 10:
            #     break
    return list_hist
 
if __name__ == "__main__":
    print("RESET INDEX")
    for path in os.listdir("./index/"):
        with open("./index/" + path,"w") as index_file:
            print("RESET " + path)
    codebook = np.loadtxt("./codebook.txt")
    print("GENERATING DATASET")
    
    y = None
    label = {}
    X = None
    child = []
    list_hist = get_all_hist_in_folder(config.paths[0], config.detector, codebook, True)  
        
    if X is None:
        X = list_hist
    else:
        X = np.append(X, list_hist, axis=0)
    
    # print(np.sum(X, axis=0))
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
