import tensorflow as tf

a = tf.constant([5])
b = tf.constant([2])

c = tf.add(a, b, "c")

with tf.Session() as session:
    result = session.run(c)
    print "The addition of this two constants is: {0}".format(result)

c = tf.multiply(a, b, "c")

matrixA = tf.constant([[2,3],[3,4]])
matrixB = tf.constant([[2,3],[3,4]])

first_operation = tf.multiply(matrixA, matrixB)
second_operation = tf.matmul(matrixA, matrixB)

with tf.Session() as session:
    result = session.run(first_operation)
    print "Element-wise multiplication: \n", result

    result = session.run(second_operation)
    print "Matrix Multiplication: \n", result