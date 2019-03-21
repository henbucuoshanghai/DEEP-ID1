import os
import numpy as np
import cv2


def get_imgs(data_pwd,out_pwd):
	if not os.path.exists(out_pwd):
		os.mkdir(out_pwd)

	for people_fd in os.listdir(data_pwd):
		people_pwd=os.path.join(data_pwd,people_fd)
		out_people_pwd=os.path.join(out_pwd,people_fd)
		if not os.path.exists(out_people_pwd):
			os.mkdir(out_people_pwd)
		for holder in os.listdir(people_pwd):
		   for img in os.listdir(os.path.join(people_pwd,holder)): 
			img_pwd=os.path.join(people_pwd,holder,img)
			out_img_pwd=os.path.join(out_people_pwd,img)	
			img=cv2.imread(img_pwd)
			h,w=img.shape[0],img.shape[1]
			out_img=img[int(h/4):int(3*h/4),int(w/4):int(3*w/4),:]  
			'''  10*3*2 patches ,choose it yourself
			'''
			cv2.imwrite(out_img_pwd,out_img)


if __name__=="__main__":
	data_pwd='/home/ubuntu/deep_id_1/all_data'
	out_pwd='/home/ubuntu/deep_id_1/out_data'
	get_imgs(data_pwd,out_pwd)
