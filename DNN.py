import tensorflow as tf
import numpy as np
FC_SIZE=160
people_num=3
CONV1_SIZE=4
NUM_CHANNELS=3  #1
CONV1_DEEP=20
CONV2_SIZE=2
CONV2_DEEP=40
CONV3_SIZE=2
CONV3_DEEP=60
CONV4_SIZE=2
CONV4_DEEP=80
def dnn(input_tensor):
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable(
			"weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
		pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

	with tf.variable_scope("layer2-conv2"):
		conv2_weights = tf.get_variable(
			"weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

 	with tf.variable_scope("layer3-conv3"):
                conv3_weights = tf.get_variable(
                        "weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
                conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
                relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
                pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("layer4-conv4"):
                conv4_weights = tf.get_variable(
                        "weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
                conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')

	with tf.variable_scope("layer3-fc"):
		pool_shape_3 = pool3.get_shape()
		nodes_3 = pool_shape_3[1] * pool_shape_3[2] * pool_shape_3[3]
		#reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
		reshaped_3 = tf.reshape(pool3, [-1, nodes_3])

	with tf.variable_scope("layer4-fc"):
                shape_4 = conv4.get_shape()
                nodes_4 = shape_4[1] * shape_4[2] * shape_4[3]
                #reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
                reshaped_4 = tf.reshape(conv4, [-1, nodes_4])

	with tf.variable_scope("mul-fc"):
		fc1_weights_3 = tf.get_variable("weight", [nodes_3, FC_SIZE],                                                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc1_weights_4 = tf.get_variable("weight1", [nodes_4, FC_SIZE],
                         initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

		deep_id=tf.nn.relu(tf.matmul(reshaped_3, fc1_weights_3) +tf.matmul(reshaped_4, fc1_weights_4)+fc1_biases)
		
	with tf.variable_scope("pre_people"):
		pre_weights = tf.get_variable("weight", [FC_SIZE, people_num],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))	
		pre_biases = tf.get_variable("bias", [people_num], initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(deep_id, pre_weights) + pre_biases

	return logit


