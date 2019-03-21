import DNN
import tensorflow as tf
import get_batch
import cv2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
people_num=3
batch_size=6
batch_num=20
if __name__=='__main__':
        out_pwd='/home/ubuntu/deep_id_1/out_data'
        train_data,valid_data,test_data=get_batch.get_batch(out_pwd)
	train_x=[]
	train_label=[]
	valid_x=[]
	valid_label=[]
	for i in train_data:
		img=cv2.imread(i[0])
		img=cv2.resize(img,(100,100))
		img=(img/225.0)*2.0-1.0
		train_x.append(img)
		train_label.append(i[1])

	for i in valid_data:
                img=cv2.imread(i[0])
                img=cv2.resize(img,(100,100))
                img=(img/225.0)*2.0-1.0
                valid_x.append(img)
                valid_label.append(i[1])
	valid_one_hot=np.zeros((len(valid_x),people_num))	
	for i,k in enumerate(valid_label):
                valid_one_hot[i][k]=1

	one_hot=np.zeros((len(train_x),people_num))
	for i,k in enumerate(train_label):
		one_hot[i][k]=1
	train_label=one_hot
	print train_label

        x_ = tf.placeholder(tf.float32, [None,100,100,3],name='x-input')
        y_ = tf.placeholder(tf.float32, [None,3], name='y-input')
	y=DNN.dnn(x_)	
	print y
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
	global_step = tf.Variable(0, trainable=False)

	correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	

	train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss,global_step=global_step)
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    tf.global_variables_initializer().run()
	    batch_indx=0
	    for i in range(50001):
       		x_batch=train_x[batch_indx*batch_size:batch_indx*batch_size+batch_size]
        	y_batch=train_label[batch_indx*batch_size:batch_indx*batch_size+batch_size]
       		batch_indx+=1
        	if  batch_indx>=batch_num:
                        batch_indx=0		

		_, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x_:x_batch, y_: y_batch})
		if i % 50 == 0 and i != 0:
			acc = sess.run(accuracy, feed_dict={x_: valid_x, y_: valid_one_hot})
			print 'train',loss_value
			print 'valid',acc
                        saver.save(sess, 'checkpoint/%05d.ckpt' % i)
