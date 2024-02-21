import numpy as np
from collections import deque
import tensorflow as tf

arr1 = np.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1],
         [1, 1, 1, 1]])
arr2 = np.array(
        [[2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2],
         [2, 2, 2, 2]])
arr3 = np.array(
        [[3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3],
         [3, 3, 3, 3]])
# print("arr1 size:", str(arr1.shape))
# stacked_arr = np.stack((arr1, arr2, arr3), axis=-1)
# print("stacked_arr size:", str(stacked_arr.shape))
# print(stacked_arr)


# Testing deque
# test = deque([np.zeros((5, 5), dtype=np.uint8) for i in range(4)], maxlen=4)
# print(test.size)

cur_stacked_images = deque(maxlen = 4)
cur_stacked_images.append(arr1)
print(cur_stacked_images)
cur_stacked_images.append(arr3)
print(cur_stacked_images)
print("State: ")
state = cur_stacked_images
# state = tf.constant(cur_stacked_images, tf.float32)

print(state)
print("test")
# state = np.expand_dims(np.array(cur_stacked_images), axis=0)
# print(state)
# next_state = np.stack(cur_stacked_images, axis = 0)
# print(next_state)