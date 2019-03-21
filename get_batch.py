import numpy as np
import os
import cv2
import random
import pickle

def get_batch(out_pwd):
	total_person=0
	total_imgs=0
	label=0
	train_data=[]
	test_pre_data=[]
	valid_data=[]
	test_data=[]
        for people_fd in os.listdir(out_pwd):
                out_people_pwd=os.path.join(out_pwd,people_fd)
		people_all_imgs=[]
                for img in os.listdir(out_people_pwd):
                       people_all_imgs.append(os.path.join(out_people_pwd,img))
		random.shuffle(people_all_imgs)
		total_person+=1
		total_imgs+=len(people_all_imgs)
		if len(people_all_imgs)<50:
			test_pre_data.append(people_all_imgs)
		else:
			train_data+=zip(people_all_imgs[5:50],[label]*45)
			valid_data+=zip(people_all_imgs[:5],[label]*5)
			label+=1
	#get train and valid
	for i,person in enumerate(test_pre_data):
		for k in range(5):
			same=random.sample(person,2)
			test_data.append((same[0],same[1],1))
		for k in range(5):
			j=i
			while j==i:
			    j=random.randint(0,len(test_pre_data)-1)
			test_data.append((random.choice(test_pre_data[i]),random.choice(test_pre_data[j]),0))
	
	random.shuffle(train_data)
	random.shuffle(valid_data)
	random.shuffle(test_data)	


	print 'total_person,total_imgs',total_person,total_imgs
	print 'valid_data',len(valid_data)
	print 'train_data',len(train_data)
	print 'test_data',len(test_data)
	return train_data,valid_data,test_data

	
if __name__=="__main__":
        out_pwd='/home/ubuntu/deep_id_1/out_data'
        train_data,valid_data,test_data=get_batch(out_pwd)
