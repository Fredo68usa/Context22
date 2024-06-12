import tensorflow as tf

# Define a tensor
x = tf.constant([[1, 2], [3, 4]])

# Sort the tensor along the first axis
x_sorted = tf.sort(x, axis=0)

# Compute the sum of the sorted tensor
total_sum = tf.math.reduce_sum(x_sorted)

# Print the result
print("Total sum:", total_sum.numpy()) # Output: Total sum: 10

