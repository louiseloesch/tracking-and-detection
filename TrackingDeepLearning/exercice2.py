import tensorflow as tf

def cnn_model_fn(features):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 64, 64, 3])
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[9, 9],
      activation=tf.nn.relu)
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  x_norm1 = tf.layers.batch_normalization(pool1, training=True)
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=x_norm1,
      filters=7,
      kernel_size=[5, 5],
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  x_norm2 = tf.layers.batch_normalization(pool2, training=True)
  # Dense Layer
  pool2_flat = tf.reshape(x_norm2, [ -1, 12 * 12 * 7])
  dense = tf.layers.dense(inputs=pool2_flat, units=1000, activation=tf.nn.relu)
  x_norm = tf.layers.batch_normalization(dense, training=True)
  # Logits Layer
  descriptor = tf.layers.dense(inputs=x_norm, units=16)
  return descriptor

def loss_pair_fn(fa,fp,f_):
  diff_pos=tf.norm(fa-fp,ord='euclidean')
  return tf.reduce_sum(tf.square(diff_pos))

def loss_triplet_fn(fa,fp,f_,m,batch_size):
  m_tens=tf.constant(m,shape=[16,batch_size],dtype=tf.float32)
  one_tens=tf.constant(1,shape=[16,batch_size],dtype=tf.float32)
  diff_neg=tf.norm(fa-f_,ord='euclidean')
  diff_pos=tf.norm(fa-fp,ord='euclidean')
  num=tf.square(diff_neg)
  denum=tf.add(tf.square(diff_pos),m_tens)
  return tf.reduce_sum(tf.nn.relu(tf.subtract(one_tens,tf.divide(num,denum))))
  
def loss_batch(descriptors,batch_size):
  loss_triplet=tf.Variable(0,dtype=tf.float32)
  loss_pair=tf.Variable(0,dtype=tf.float32)
  m=0.01
  #arr[:,0:20:3]
  fa=descriptors[0:batch_size:3]
  fp=descriptors[1:batch_size:3]
  f_=descriptors[2:batch_size:3]
  loss_pair=loss_pair_fn(fa,fp,f_)
  loss_triplet=loss_triplet_fn(fa,fp,f_,m,batch_size)
  
  return tf.add(loss_pair,loss_triplet)
