import Adafruit_PCA9685 as pca
import numpy as np
import time
import subprocess

max_arr0 = []
min_arr0 = []
max_arr1 = []
min_arr1 = []

# first 3 are pwm0, second 3 are pwm1

class Servo(object):


	def __init__(self, freq):
		self.pwm0 = pca.PCA9685(0x40)
		self.pwm1 = pca.PCA9685(0x41)
		self.pwm0.set_pwm_freq(freq)
		self.pwm1.set_pwm_freq(freq)

		self.servo_num = [
						0, 2, 1, 3, 4, 6, 10, 7, 9, # first 3
						0, 1, 2, 3, 4, 10, 6, 7, 8] # second 3

		self.home = np.array([[
					1600, 1350, 1600, 1600, 1350, 1750, 1650, 1850, 1550, # first 3
					1600, 1350, 1550, 1600, 1350, 1650, 1600, 1850, 1550]]).astype('float32') # second 3


	def apply(self, actions):
		x = 0
		for a in actions[0]:
			if x < 9:
				while True:
					try:
						self.pwm0.set_pwm(self.servo_num[x], 0, int(a))
						break
					except IOError:
						# account for i2c connection errors
						subprocess.call(['i2cdetect', '-y', '1'])
						print('reset')

			elif x < len(self.servo_num) and x > 8:
				while True:
					try:
						self.pwm1.set_pwm(self.servo_num[x], 0, int(a))
						break
					except IOError:
						# account for i2c connection errors
						subprocess.call(['i2cdetect', '-y', '1'])
						print('reset')
			x += 1

	'''
	def sig_to_angle(self, sig):
		return ((sig * 1400) / 165) + 800


	def verify(self, a1, a2):
		# for 1,8 (a2) + 0,2 (a1)
		a1 = sig_to_angle(a1)
		a2 = sig_to_angle(a2)
		# relationship between two servos
		collis = -151.84 + (67.89 * math.log(a1, math.e))

		col2 = -58.35227 + (49.49654 * math.log(angle(a), math.e))
	'''

	def reset(self):
		self.apply(self.home)

		return self.home

