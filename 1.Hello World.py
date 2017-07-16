import tensorflow as tf

# Constants
a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a, b)

with tf.Session() as session:
    result = session.run(c)
    print(result)

# Variables
state = tf.Variable(0)

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init_op)
    print(session.run(state))
    for _ in range(3):
        session.run(update)
        print(session.run(state))

# Placeholders
a = tf.placeholder(tf.float32)
b = tf.multiply(a, 2)

with tf.Session() as sess:
    result = sess.run(b, feed_dict={a:3.5})
    print(result)