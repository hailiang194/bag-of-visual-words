import cv2
import sys
import config
import numpy as np
import random
import imutils
import predict
import time

if __name__ == "__main__":
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    image = cv2.imread(sys.argv[1])
    resized = imutils.resize(image, width= 35) 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(resized)
    ss.switchToSelectiveSearchFast()

    code_book = np.loadtxt("./codebook.txt")
    weight = np.loadtxt("./weight.txt")
    database_vector = np.loadtxt("./query_vector.txt")
    image_path = []
    clock = time.process_time() 

    for x, y, w, h in ss.process():
        if 1000 > w * h >  500:
            # print(w * h)
            clone = resized.copy()
            orinal_clone = image.copy()
            color = [random.randint(0, 255) for i in range(0, 3)]
            cv2.rectangle(clone, (x, y), (x + w, y + h), color , 1) 
            
            query_x, query_y, query_w, query_h = (x * image.shape[0] // resized.shape[0], y * image.shape[1] // resized.shape[1], w * image.shape[0] // resized.shape[0], h * image.shape[1] // resized.shape[1])

            cv2.rectangle(orinal_clone, (query_x, query_y), (query_x + query_w, query_y + query_h), color , 3) 
            scores, accepted_score = predict.get_score(image[query_y:query_y + query_h, query_x:query_x + query_w], config.detector, code_book, weight, database_vector, top=5)
            # scores, accepted_score = predict.get_score(resized[x:x+w,y:y+h], code_book, weight, database_vector)             

            for score, child_path in predict.get_query_image("../books/", scores, accepted_score):
                full_path = "../books/" + child_path
                # print(full_path)
                image_path.append(full_path)
            # print()
            # cv2.imshow("image", clone)
            # cv2.imshow("original",image[query_y:query_y + query_h, query_x:query_x + query_w])
            # key = cv2.waitKey(0)
            # if key == ord('q'):
                # break
    print(time.process_time() - clock)
    print((sorted(set(image_path))))
