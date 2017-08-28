import os
import cv2
import torch
import numpy as np
from torch.multiprocessing import Pool

import sys
sys.path.append('yolo/')

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg


def preprocess(camera):
	_, image = camera.read()
	im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.inp_size))[0], 0)
	return image, im_data


class CNN(object):

	def __init__(self, thresh=0.1, width=600, height=480):

		self.center = (width/2, height/2)
		self.camera = cv2.VideoCapture(0)
		self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		self.max_area = width*height

		trained_model = cfg.trained_model
		self.thresh = thresh

		self.net = Darknet19()
		net_utils.load_net(trained_model, self.net)
		self.net.cuda()
		self.net.eval()


	def _area(self, t_left, b_right):
		x0, y0 = t_left
		x1, y1 = b_right
		length = x1 - x0
		width = y0 - y1

		return abs(length*width)


	def _difference(self, t_left, b_right):
		box_center = (int((t_left[0]+b_right[0])/2), int((t_left[1]+b_right[1])/2))
		diff = (self.center[0] - box_center[0], self.center[1] - box_center[1])

		return diff[0]


	def process(self):
		while True:
			image, im_data = preprocess(self.camera)
			im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
			bbox_pred, iou_pred, prob_pred = self.net(im_data)

			bbox_pred = bbox_pred.data.cpu().numpy()
			iou_pred = iou_pred.data.cpu().numpy()
			prob_pred = prob_pred.data.cpu().numpy()

			bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, self.thresh)
			out = np.ones((1,2)).astype('float32')

			for x in range(len(bboxes)):
				if cls_inds[x] == 14:
					topleft = (bboxes[x][0], bboxes[x][1])
					bottomright = (bboxes[x][2], bboxes[x][3])
					conf = scores[x]
					detect = True

					diff = self._difference(topleft, bottomright)
					area = self._area(topleft, bottomright)

					out[0][0] = diff
					out[0][0] /= self.center[0]
					out[0][1] = area
					out[0][1] /= self.max_area

					return out


	def quit(self):
		self.camera.release()



