#!/usr/bin/env python3
# -*- encoding: iso-8859-1 -*-

import sys
import rospy
import roslib
import rospkg

import numpy as np
import cv2
import random
import imutils

class c_data_augmentation:

	def img_debug(self, img):
		cv2.imshow("Debug", img)
		cv2.waitKey(0)

	def fill(self, img, h, w):
		img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
		return img

	def noise_shift(self, img, noise_sigma):
		h, w = img.shape[:2]
		noise = np.random.randn(h, w) * noise_sigma
		img[:,:,0] = img[:,:,0] + noise
		img[:,:,1] = img[:,:,1] + noise
		img[:,:,2] = img[:,:,2] + noise
		return img

	def rotation_ccw_shift(self, img, angle):
		angle = int(random.uniform(1, angle))
		h, w = img.shape[:2]
		M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
		img = cv2.warpAffine(img, M, (w, h))
		return img

	def rotation_cw_shift(self, img, angle):
		angle = int(random.uniform(-angle, 1))
		h, w = img.shape[:2]
		M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
		img = cv2.warpAffine(img, M, (w, h))
		return img

	def vertical_down_shift(self, img, ratio=0.0):
		ratio = random.uniform(0, ratio)
		w, h = img.shape[:2]
		to_shift = w*ratio
		img = img[:int(w-to_shift), :, :]
		img = self.fill(img, h, w)
		return img

	def vertical_up_shift(self, img, ratio=0.0):
		ratio = random.uniform(-ratio, 0)
		w, h = img.shape[:2]
		to_shift = w*ratio
		img = img[int(-1*to_shift):, :, :]
		img = self.fill(img, h, w)
		return img

	def horizontal_right_shift(self, img, ratio=0.0):
		ratio = random.uniform(0, ratio)
		w, h = img.shape[:2]
		to_shift = w*ratio
		img = img[:, :int(h-to_shift), :]
		img = self.fill(img, h, w)
		return img


	def horizontal_left_shift(self, img, ratio=0.0):
		ratio = random.uniform(-ratio, 0)
		w, h = img.shape[:2]
		to_shift = w*ratio
		img = img[:, int(-1*to_shift):, :]
		img = self.fill(img, h, w)
		return img

	def update_data_base(self, img, img_name, img_path, label):
		#Salva imagem_hr
		cv2.imwrite(img_path + img_name , img)
		#Adiciona nova informação no arquivo de controle
		self.train_file.write(img_name + " " + label + "\n")


	def __init__(self, _horizontal_left, _horizontal_right, _vertical_up, _vertical_down, _rotation_cw, _rotation_ccw, _noise):

		self.horizontal_left = _horizontal_left
		self.horizontal_right = _horizontal_right
		self.vertical_up = _vertical_up
		self.vertical_down = _vertical_down
		self.rotation_cw = _rotation_cw
		self.rotation_ccw = _rotation_ccw
		self.noise = _noise

		#Carrega endereco do package omni_ros_gui
		rospac = rospkg.RosPack()
		self.ml_lab3_path = rospac.get_path("ml_lab3")
		self.ml_lab3_path += "/scripts"

		train_path =  self.ml_lab3_path + '/meses/train_test.txt'

		print ("Carregando arquivo das imagens...")
		#Abre arquivo para leitura
		self.train_file = open(train_path, 'r+')
		texto = self.train_file.read()
		train_paths = texto.split('\n')

		#Remove elemento não esperado no vetor
		train_paths.remove('')
		#train_paths.sort()

		print ("Loading training set...")
		x = []
		y = []
		for image_path in train_paths:
			image_name, label = image_path.split(' ')
			## Image path
			path = self.ml_lab3_path + "/meses/data_test/"
			img = cv2.imread(path + image_name)

			if self.horizontal_left:
				#Realiza horizontal left shift
				img_horizontal_left = self.horizontal_left_shift(img, 0.2)
				#Atualiza data base
				self.update_data_base(img_horizontal_left, image_name.split(".")[0]+"_hl.jpg", path, label)

			if self.horizontal_right:
				#Realiza horizontal right shift
				img_horizontal_right = self.horizontal_right_shift(img, 0.2)
				#Atualiza data base
				self.update_data_base(img_horizontal_right, image_name.split(".")[0]+"_hr.jpg", path, label)

			if self.vertical_up:
				#Realiza vertical up shift
				img_vertical_up = self.vertical_up_shift(img, 0.1)
				#Atualiza data base
				self.update_data_base(img_vertical_up, image_name.split(".")[0]+"_vu.jpg", path, label)

			if self.vertical_down:
				#Realiza vertical down shift
				img_vertical_down = self.vertical_down_shift(img, 0.1)
				#Atualiza data base
				self.update_data_base(img_vertical_down, image_name.split(".")[0]+"_vd.jpg", path, label)

			if self.rotation_cw:
				#Realiza rotação no sentido horário
				img_rotation_cw = self.rotation_cw_shift(img, 10)
				#Atualiza data base
				self.update_data_base(img_rotation_cw, image_name.split(".")[0]+"_cw.jpg", path, label)

			if self.rotation_ccw:
				#Realiza rotação no sentido anti-horário
				img_rotation_ccw = self.rotation_ccw_shift(img, 10)
				#Atualiza data base
				self.update_data_base(img_rotation_ccw, image_name.split(".")[0]+"_ccw.jpg", path, label)

			if self.noise:
				img_noise = self.noise_shift(img, 0.5)
				#Atualiza data base
				self.update_data_base(img_noise, image_name.split(".")[0]+"_n.jpg", path, label)


	def __del__(self):
		#Destroi janelas de visualização
		print ("Apagando janelas...")
		cv2.destroyAllWindows()
		print ("Done...")

def main(args):
	#Modifica as imagens
	cda = c_data_augmentation(True, True, True, True, True, True, False)
	#Aplica ruído em todas as imagens
	#cda = c_data_augmentation(False, False, False, False, False, False, True)
	rospy.init_node('n_data_augmentation', anonymous=True)
	try:
		print ("Doing...")
		#rospy.spin()
	except KeyboardInterrupt:
        	rospy.loginfo("Shutting down")

if __name__ == "__main__":
        main(sys.argv)
