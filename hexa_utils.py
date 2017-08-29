import Adafruit_PCA9685 as pca
import time

freq = 245

def reset():
	pwm1 = pca.PCA9685(0x41)
	pwm0.set_pwm_freq(freq)
	pwm0 = pca.PCA9685(0x40)
	pwm1.set_pwm_freq(freq)


#pwm0 center: 1450, 1450, 1650
#pwm1 center: 1650, 1450, 1450

pwm1 = pca.PCA9685(0x41)
pwm0 = pca.PCA9685(0x40)
pwm0.set_pwm_freq(freq)
pwm1.set_pwm_freq(freq)

max = 2250
min = 800
s0 = [min+800, max-900, 2200, 800]
s1 = [1600, 1850, 800, 2100]
s2 = [1650, 1850, 800, 2200]

print('done')

def set0(arr, num, pos):

	num2 = num+1

	if num == 6:
		num = 9
	else:
		num = num


	if pos == 1:
		pwm0.set_pwm(num, 0, arr[2])
		pwm0.set_pwm(num2, 0, arr[3])
	else:
		print('set0')
		print(num, num2)
		pwm0.set_pwm(num, 0, arr[0])
		pwm0.set_pwm(num2, 0, arr[1])

def set1(arr, num, pos):
	#if num == 5:
	#	temp = 9
	#else:
	#	temp = num

	#temp = num
	if pos == 1:
		pwm1.set_pwm(num, 0, arr[2])
		pwm1.set_pwm(num+1, 0, arr[3])
	else:
		print('set1')
		print(num, num+1)
		pwm1.set_pwm(num, 0, arr[0])
		pwm1.set_pwm(num+1, 0, arr[1])

def vert(num, dir):
	if num ==0:
		set0(s0, 0, dir)
		set0(s0, 3,dir)
		set0(s2, 6, dir)
	elif num == 1:
		set1(s0, 0, dir)
		set1(s0, 3, dir)
		set1(s1, 6, dir)

def move(num):
	if num == 0:
		pwm0.set_pwm(8, 0, 1850)
		pwm0.set_pwm(5, 0, 1400)
		pwm0.set_pwm(2, 0, 1150)
	elif num == 1:
		pwm1.set_pwm(2, 0, 1700)
		pwm1.set_pwm(10, 0, 1650)
		pwm1.set_pwm(8, 0, 1300)

def back(num):
	if num == 0:
		pwm0.set_pwm(2, 0, 1450)
		pwm0.set_pwm(5, 0, 1750)
		pwm0.set_pwm(8, 0, 1550)
	elif num == 1:
		pwm1.set_pwm(10, 0, 1650)
		pwm1.set_pwm(2, 0, 1300)
		pwm1.set_pwm(8, 0, 1550)

def home():
	vert(0,0)
	vert(1,0)
	back(1)
	back(0)

def walk(delay):
	vert(0,1)
	time.sleep(delay)
	move(0)
	time.sleep(delay)
	vert(0,0)
	time.sleep(delay)
	vert(1,1)
	time.sleep(delay)
	back(0)
	time.sleep(delay)
	move(1)
	time.sleep(delay)
	vert(1,0)
	time.sleep(delay)
	vert(0,1)
	time.sleep(delay)
	back(1)
	time.sleep(delay)

