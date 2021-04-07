import cv2
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.cluster.vq import vq
import sys
import numpy as np
import config
import cv2
import config
import numpy as np
from scipy.cluster.vq import vq
import sys
import os
import imutils

def get_image_hist(image, detector, code_book):
    clone_image = image.copy()
    # if len(image.shape) > 2 and image.shape[-1] == 3:
    #     clone_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    clone_image = config.pre_process_image(clone_image)
    _, desc = detector.detectAndCompute(clone_image, None)
    labels, _ = vq(desc, code_book) 

    hist = np.zeros((1, code_book.shape[0]))
    for label in labels.tolist():
        hist[0, label] += 1

    hist = hist / np.sum(hist)
    return hist

def get_score(image, detector, code_book, weight, database_vector, top=10):
    # clone = imutils.resize(image, height=500)
    clone = image.copy()

    hist = get_image_hist(clone, detector,code_book)

    vector = np.multiply(hist, weight).reshape(code_book.shape[0])
    score = np.zeros((1, database_vector.shape[0]))
    for i in range(database_vector.shape[0]):
        data = database_vector[i, :]
        score[0, i] = np.linalg.norm(data / np.linalg.norm(data) - vector / np.linalg.norm(vector))
    accepted_score = np.sort(score)[0, top]
    return score, accepted_score

def get_query_image(folder, score, accepted_score):
    for i, child_path in enumerate(os.listdir(folder)):
        if score[0, i] <= accepted_score:
            yield score[0, i], child_path
def noise(image):
    uniform = np.zeros(image.shape, np.uint8)
    cv2.randn(uniform, (0), (99))
    return cv2.add(image, uniform)

def get_matching_image(image, code_book, detector, weight, database_vector, top=10):
    hist = get_image_hist(image, detector, code_book)
    vector = np.multiply(hist, weight).reshape(code_book.shape[0])

    list_images = None
    with open("./index.txt", "r") as index_file:
        list_images = index_file.readlines()

    dot_product = hist @ database_vector.T
    score = (dot_product / np.multiply(np.linalg.norm(hist), np.linalg.norm(database_vector, axis=1))).reshape(-1)

    accepted_score = sorted(score)[-top]

    matching_images = {}
    for i in range(len(list_images)):
        if score[i] >= accepted_score:
            matching_images[list_images[i]] =  score[i]

    return matching_images

def get_accurate(image_path, matching_images):
    get_image_post = lambda path: int(path.split('/')[-1].split('.')[0][-5:])
    image_pos = get_image_post(image_path)
    print(image_path, end=",")
    start_correct_pos = image_pos // 4 * 4
    end_correct_pos = start_correct_pos + 4

    match = 0
    for matching_path in matching_images.keys():
        matching_pos = (get_image_post(matching_path))
        if start_correct_pos <= matching_pos <= end_correct_pos:
            print(matching_path.strip(), end=',')
            match += 1
        else:
            print(",", end="")
    print(match)

if __name__ == "__main__":
    # image = cv2.imread(sys.argv[1]) 
    # image = noise(image)
    # print(image)
    # image = imutils.resize(image, height=475)
    # cv2.imshow("Query", image)
    code_book = np.loadtxt("./codebook.txt")
    weight = np.loadtxt("./weight.txt")
    # hist = get_image_hist(image, cv2.SIFT_create(),code_book)
    # vector = np.multiply(hist, weight).reshape(code_book.shape[0])
    # print(vector)
    # distance_vector = np.linalg.norm(vector)
    database_vector = np.loadtxt("./query_vector.txt")
    with open("./index.txt", "r") as index_file:
        for path in index_file.readlines():
            image = cv2.imread(config.paths[0] + path.strip())
            matching_images = get_matching_image(image, code_book, config.detector, weight, database_vector, 20)
            get_accurate(path.strip(), matching_images)
    # print(matching_image)
    # hist = get_image_hist(image, config.detector, code_book)
    # vector = np.multiply(hist, weight).reshape(code_book.shape[0])

    # list_images = None
    # with open("./index.txt", "r") as index_file:
    #     list_images = index_file.readlines()

    # np.savetxt(sys.stdout, vector, "%.2f")
    # index_images = []
    # for index in range(vector.shape[0]):
    #     if vector[index] > :
    #         print(vector[index])
    #         for image_index in range(database_vector[:, index].shape[0]):
    #             if database_vector[image_index, index] > 4e-3:
    #                 index_images.append(image_index)
    # index_images = {}
    # count = 0
    # for index in range(database_vector.shape[0]):
    #     total_match = np.count_nonzero(np.logical_and(vector > 0, database_vector[index, :] > 0))
    #     if total_match >=  70:
    #         print(list_images[index].strip(), end=', ') 
    #         count += 1

    # print('\n{}'.format(count))
    # print(len(set(index_images)))
    # print([list_images[index] for index in index_images])
    # dot_product = hist @ database_vector.T
    # score = (dot_product / np.multiply(np.linalg.norm(hist), np.linalg.norm(database_vector, axis=1))).reshape(-1)
    # print(score[1])
    # print("MAX=" + str(np.amax(score)))
    # accepted_score = (sorted(score.tolist())[-70])
    # score = []
    # for i in range(database_vector.shape[0]):
    #     score.append(np.linalg.norm(vector / np.linalg.norm(vector) - database_vector[i, :] / np.linalg.norm(database_vector[i, :])))
    # score = np.array(score)
    # count = 0
    # filtered_score = []
    # mapping_index = []
    # for i in range(score.shape[0]):
        # total_match = np.count_nonzero(np.logical_and(vector > 0, database_vector[i, :] > 0))
        # if (np.count_nonzero(vector > 0) >= 2 * total_match or np.count_nonzero(vector > 0) == total_match):
        # if np.count_nonzero(vector > 0) - total_match < np.count_nonzero(vector > 0) * 1.00 and score[i] >= np.mean(score):
        #    filtered_score.append(score[i])
        #    mapping_index.append(i)
            # print(total_match, np.count_nonzero(vector > 0), list_images[i].strip(), sep=',')
            # count += 1
    #accepted_score = sorted(filtered_score)[-10]
    # print(filtered_score)
    # accepted_score = np.amax(filtered_score) - np.mean(filtered_score)
    #accepted_score = np.amax(filtered_score) - (np.amax(filtered_score) - np.amin(filtered_score)) * 0.60
    #print(accepted_score, len(filtered_score))
    # print(score[-2])
    #         searched_image = cv2.imread("../books/" + list_images[i].strip())
    #         cv2.imshow("Searched", searched_image)
    #         cv2.waitKey(0)
    # print(score)
    # searched_image = set()
    # for i in range(vector.shape[0]):
    #     if vector[i] > 0:
    #         # print(vector[i])
    #         included_images = np.loadtxt("./index/" + str(i) + ".txt", dtype=int)
    #         print(i)
    #         # print(included_images.shape)
    #         searched_image = searched_image.union(included_images.tolist())

    # print(searched_image)
    # print(len(searched_image))
    # score = np.zeros((len(list_images)))
    # for i in range(len(list_images)):
    #     score[i] = np.sum(np.logical_and(vector>0, database_vector[i, :] > 0).astype(int))
    # print(2 - 2 * (vector @ database_vector[i, :]))
    # score =  ((vector > 0).astype(int) @ (database_vector > 0).astype(int).T)
    # score = np.all((vector > 0), (database_vector > 0))
    # score = (vector > 0) | (database_vector > 0)
    # print(score) 
    # print(sorted(score.tolist())[-10])
    # score = np.abs(vector - database_vector)
    # score = (np.sum(score, axis=1))
    # score = 2 - 2 * np.sum(np.multiply(vector, database_vector), axis=1)
    # print(score)
    # accepted_score = sorted(score.tolist())[50]
    # print(", ".join([(list_images[i].strip() +  str(score[i])) for i in mapping_index]))
    # print(mapping_index)
    # for i in range(len(mapping_index)):
    #    if filtered_score[i] >= accepted_score:
    #        print(list_images[mapping_index[i]], filtered_score[i])
            #searched_image = cv2.imread("../books/" + list_images[mapping_index[i]].strip())
            #cv2.imshow("Test", searched_image)
            #key = cv2.waitKey(0)
            #if key == ord('q'):
            #    break
    # # print(database_vector)
    # score = np.zeros((1, database_vector.shape[0]))
    # for i in range(database_vector.shape[0]):
    #     data = database_vector[i, :]
    #     distance_data = np.linalg.norm(data)
    #     score[0, i] = np.linalg.norm(data - vector)

    # print(score[0, 0])
    # score, accepted_score = get_score(image, config.detector, code_book, weight, database_vector)

    # accepted_score = np.sort(score)[0, 20]
    # # for i, path in enumerate(os.listdir("../books/")):
    #     # if score[0, i] <= accepted_score:
    # for s, path in get_query_image("../books/", score, accepted_score):

    #     if True:
    #         expected = " Expected" if "../books/" + path in sys.argv[2:]  else ""
    #         print(str(s) + expected)
    #         mapping_image = cv2.imread("../books/" + path)
    #         cv2.imshow("Mapping", mapping_image)
    #         key = cv2.waitKey(0)
    #         if key == ord('q'):
    #             break
    #         elif len(expected) > 0:
    #             break
    # # cv2.waitKey(0)
