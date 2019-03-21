from DNN import dnn
import tensorflow as tf
import get_batch
import cv2
import numpy as np
import scipy
from scipy.spatial.distance import cosine
people_num=3
batch_size=6
batch_num=20
if __name__=='__main__':
        out_pwd='/home/ubuntu/deep_id_1/out_data'
        train_data,valid_data,test_data=get_batch.get_batch(out_pwd)
	test_x1=[]
	test_x2=[]
	test_label=[]
	for i in test_data:
		img=cv2.imread(i[0])
		img=cv2.resize(img,(100,100))
		img=(img/225.0)*2.0-1.0
		test_x1.append(img)

		img=cv2.imread(i[1])
                img=cv2.resize(img,(100,100))
                img=(img/225.0)*2.0-1.0
                test_x2.append(img)

		test_label.append(i[2])

	print test_label
	
        x_ = tf.placeholder(tf.float32, [None,100,100,3],name='x-input')
        y_ = tf.placeholder(tf.float32, [None,3], name='y-input')
	y=dnn(x_)	
	print y

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,"zuotiande/49000.ckpt")
	        id_1=sess.run(y,{x_: test_x1})	
		id_2=sess.run(y,{x_: test_x2})
		print id_1[0]
		
		pre_y = [cosine(x, y) for x, y in zip(id_1,id_2)]
		print pre_y
		pre_y=[1 if pre_y[i]>=0.5 else 0  for i in range(len(pre_y))]
		cor=0
		for i in range(len(pre_y)):
			if pre_y[i]==test_label[i]:
				cor+=1
		acc=1.0*cor/len(pre_y)	
		print acc
		print pre_y
		print test_label

