import numpy as np
import tensorflow.compat.v1 as tf

point = np.loadtxt("./desc.txt", dtype=np.float32)
desc_num, dimestion = point.shape

# input_fn = tf.train_limit_epochs(tf.convert_to_tensor(point, dtype=tf.float32), num_epochs=1)
num_steps = 1
batch_size = 1024
k = 1000

# num_cluster = 1000
# kmeans = tf.estimator.experimental.KMeans(num_cluster=num_cluster, use_mini_batch=False)
