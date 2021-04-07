import numpy as np
import sys

if __name__ == "__main__":
    database_hist = np.loadtxt("./all_hist.txt")
    print(database_hist.shape)
    existed = (database_hist > 0).astype(int)
    np.savetxt(sys.stdout, np.sum(existed, axis=0), fmt="%d", newline=' ')
    weight = np.log(existed.shape[0] / np.sum(existed, axis=0))
    np.savetxt("./weight.txt", weight)
    # print(weight.shape)
    query_vector = np.multiply(weight, database_hist)
    np.savetxt("./query_vector.txt", query_vector)
