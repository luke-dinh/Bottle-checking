import cv2
import os, sys
import DateTime as dt
import numpy as np
import configparser
import matplotlib.pyplot as plt
from PIL import Image
import imutils
from pylab import *
import argparse
from scipy.spatial import distance as dist

# This is configuration for ROI
config = configparser.ConfigParser()
config.read("system.ini")
IMAGES_MASK = config.get("path", "images_mask")
SOURCE_IMAGES_PATH = config.get("path", "source_dir")
RESULT_IMAGES_PATH = config.get("path", "result_dir")
CUT_BORDER = config.getfloat("geometry", "cut_border")
PRE_ROI_WIDTH = config.getfloat("geometry", "pre_roi_width")
ROI_WIDTH = config.getfloat("geometry", "roi_width")

# Threshold to detect label ( relative)
LABEL_H_MIN = 0
LABEL_H_MAX = 250
LABEL_S_MIN = 142
LABEL_S_MAX = 255
LABEL_V_MIN = 106
LABEL_V_MAX = 255

# threshold for liquid level
LIQUID_LEVEL_H_MIN = 0
LIQUID_LEVEL_H_MAX = 255
LIQUID_LEVEL_S_MIN = 0
LIQUID_LEVEL_S_MAX = 111
LIQUID_LEVEL_V_MIN = 0
LIQUID_LEVEL_V_MAX = 111

# 
BANG_HEIGHT = 0.2

#
def input_image(path):
	image = cv2.imread(path,1)
	image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
	image_e = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image
#
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
#
def his_enhance(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.equalizeHist(image)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	return image
#

def get_roi(image, left_border, right_border):
    image_h, image_w, _ = image.shape
    left_part = image[:, 0:left_border]
    right_part = image[:, right_border:image_w]
    return np.column_stack((left_part, right_part))

#------------------------------------------------------------------------------------------------------------

def find_background(image, roi):
    image_h, image_w, _ = image.shape
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # plt.imshow(roi_hist)
    # plt.show()
    mask = cv2.calcBackProject([hsv_img], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    # cv2.imshow('Mask ',mask)
    ksize = int(0.005 * image_h)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.filter2D(mask, -1, kernel)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask
#-------------------------------------------------------------------------------------------------------------

def get_pre_borders(mask):
	components = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
	_, labelmap, stats, centers = components
	# print('stats matrix( get pre border)',stats)
	st = stats[:, 3]
	largest = np.argmax(st)
	# print('Index of largest label \n', largest)
	st[largest] = 0
	second = np.argmax(st)
	# print('Index of second largest label \n', second)
	left = stats[second, 0]
	width = stats[second, 2]
	right = left + width
	roi_width = int(width * ROI_WIDTH)
	return left, right, roi_width

# --------------------------------------------------------------------------------------------------------------
def ConnectedComponents(bin1):

	components = cv2.connectedComponentsWithStats(bin1, connectivity=8, ltype=cv2.CV_32S)
	_, labelmap, stats, centers = components
	st = stats[:, 4]
	st_y = stats[:, 1]
	# print('Test stat1',stats)
	S_largests = [np.argmax(st)]
	# print('largests1',S_largests)
	# st[S_largests[0]] = 0
	S1000 = np.where(st > 1000)
	st_arg = S1000[0]
	# print('st argument',st_arg)
	# bin_h,bin_w =bin.shape
	number_label = len(st_arg)
	# print('number of label(func)1:',number_label)
	if (len(st_arg) > 2):
		# Bang /lid area
		# print('st matrix',st)
		S_smallest_arg = np.argmin(st)
		# print('Smallest area index', S_smallest_arg)
		left_bang = stats[S_smallest_arg,0]
		top_bang = stats[S_smallest_arg,1]
		right_bang = left_bang + stats[S_smallest_arg,2]
		bottom_bang = top_bang + stats[S_smallest_arg,3]
		mask_bin = np.zeros(bin1.shape, dtype=np.uint8)
		# liquid level in axis y => max
		# print('st_y matrix',st_y)
		y_max_arg = np.argmax(st_y)
		# print('Largest y index', y_max_arg)
		left_liq = stats[y_max_arg,0]
		# print('left liquid',left_liq)
		top_liq = stats[y_max_arg,1]
		# print('top liquid',top_liq)
		right_liq = left_liq + stats[y_max_arg,2]
		# print('right liquid',left_liq)
		bottom_liq = top_liq + stats[y_max_arg,3]
		# print('bottom liquid',left_liq)	
		mask_bin[top_bang:bottom_bang,left_bang:right_bang] = 255
		mask_bin[top_liq:bottom_liq,left_liq:right_liq] = 255

	return number_label, mask_bin
#---------------------------------------------------------------------------------------------------------

def get_bottle_mask(bin):
	def clean(cln_bin, larg_num):
		components = cv2.connectedComponentsWithStats(cln_bin, connectivity=8, ltype=cv2.CV_32S)
		_, labelmap, stats, centers = components
		st = stats[:, 4]
		# print('stats matrix (get bottle mask)',stats)
		largests = [np.argmax(st)]
		st[largests[0]] = 0
		largests.append(np.argmax(st))
		cln_bin = np.zeros(cln_bin.shape, dtype=np.uint8)
		cln_bin[labelmap == largests[larg_num]] = 255
		return cln_bin, stats[largests[0]]
 
	bin, _ = clean(bin, 1)
	bin = cv2.bitwise_not(bin)
	bin, stats = clean(bin, 0)
	left = stats[0]
	top = stats[1]
	right = left + stats[2]
	bottom = top + stats[3]
	mask = bin[top:bottom, left:right]
	mask = cv2.merge((mask, mask, mask))
	return mask, left, top, right, bottom

#------------------------------------------------------------------------------------------------------
# The liquid level segmantation
def liqlevel_segmentation(image):
	image_h, image_w, _ = image.shape
	center_img = image[:, int(image_w * 0.03): int(image_w * 0.97)]
	blur = int(0.015 * image_w)
	center_h, center_w, _ = center_img.shape
	img_hsv = cv2.cvtColor(center_img, cv2.COLOR_RGB2HSV)
	filtered = cv2.inRange(img_hsv, (LIQUID_LEVEL_H_MIN, LIQUID_LEVEL_S_MIN, LIQUID_LEVEL_V_MIN),(LIQUID_LEVEL_H_MAX, LIQUID_LEVEL_S_MAX, LIQUID_LEVEL_V_MAX))
	binary = cv2.GaussianBlur(filtered, (5,5), 15)
	ret, binary = cv2.threshold(binary,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	binary = cv2.GaussianBlur(filtered, (15,15), 15)
	ret, binary = cv2.threshold(binary,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	binary = cv2.GaussianBlur(filtered, (15,15), 15)
	ret, binary = cv2.threshold(binary,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
	binary =cv2.bitwise_not (binary)
	return binary

# ----------------------------------------------------------------------------------------------------
# Label segmentation
def label_segmentation(image):
	image_h, image_w, _ = image.shape
	center_img = image[:, int(image_w * 0.03): int(image_w * 0.97)]
	blur = int(0.015 * image_w)
	center_h, center_w, _ = center_img.shape
	img_hsv = cv2.cvtColor(center_img, cv2.COLOR_RGB2HSV)
	filtered = cv2.inRange(img_hsv, (LABEL_H_MIN, LABEL_S_MIN, LABEL_V_MIN),(LABEL_H_MAX, LABEL_S_MAX, LABEL_V_MAX))
	binary = cv2.GaussianBlur(filtered, (5,5), 15)
	ret, binary = cv2.threshold(binary,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	binary = cv2.GaussianBlur(filtered, (15,15), 15)
	ret, binary = cv2.threshold(binary,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	binary = cv2.GaussianBlur(filtered, (15,15), 15)
	ret, binary = cv2.threshold(binary,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	binary =cv2.bitwise_not (binary)
	return binary

# ----------------------------------------------------------------------------------------------------
# Check the existence of label
def check_label_exist(image, S_good_label):
	binary = label_segmentation(image)
	cv2.imshow('binary-check_label_exist',binary)
	contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print('Length of contours-testbefore: check_lablels_exist()',len(contours))
	try:
		if (len(contours) >= 2):
			c = max(contours, key = cv2.contourArea)
			contours.remove(c)
			while (len(contours) >= 2):
				c1 = min(contours, key = cv2.contourArea)
				contours.remove(c1)
	except:
		print('error check label exists')
	print('Length of contours-testafter: check_lablels_exist()',len(contours))
	cnt = contours[0]
	S_label = int(cv2.contourArea(cnt))
	if (S_label < (S_good_label / 2)):
		label_exist = False
	else:
		label_exist = True
	return label_exist

# ------------------------------------------------------------------------------------------------------
# Find the height of label (y_axis)
def find_labels(image):
	image_h, image_w, _ = image.shape
	center_img = image[:, int(image_w * 0.03): int(image_w * 0.97)]
	binary = label_segmentation(image)
	cv2.imshow('binary-find_labels',binary)
	contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print('Length of contours-testbefore: find_lablels()',len(contours))
	try:
		if (len(contours) >= 2):
			c = max(contours, key = cv2.contourArea)
			contours.remove(c)
			while (len(contours) >= 2):
				c1 = min(contours, key = cv2.contourArea)
				contours.remove(c1)
	except:
		print('error find label')
	print('Length of contours-testafter: find_lablels()',len(contours))
	binary = cv2.drawContours(binary, contours, -1, (100,87,51),3)
	hull = [cv2.convexHull(c) for c in contours]
	binary = cv2.drawContours(binary,hull,-1,(100,87,55),3)
	# cv2.imshow('binary label-find_labels',binary)
	label_y_arr = []
	for num, cnt in enumerate(contours):
		x, y, w, h = cv2.boundingRect(cnt)
        # Getting ROI 
		roi = image[y:y+h, x:x+w] 
		# show ROI 
		# cv2.imshow('segment label no:'+str(num),roi) 
		cv2.rectangle(center_img,(x,y),( x + w, y + h ),(255,0,0),2)
		cv2.imshow('marked label areas',center_img) 
		if (y > (BANG_HEIGHT * image_h)):
			label_y_arr.append(y)
			label_y_arr.append(y + h)
	min_label_y, max_label_y = 0, 0
	if label_y_arr:
		min_label_y = min(label_y_arr)
		max_label_y = max(label_y_arr)
	return min_label_y, max_label_y

#----------------------------------------------------------------------------------------------------------
# Find the exact ratio between label and bottle
def right_labels():
	image = input_image('D:\\Testimg\\Bottle_checking (CC)\\good1.jpg')
	binary = label_segmentation(image)
	image_h, image_w, _ = image.shape
	center_img = image[:, int(image_w * 0.03): int(image_w * 0.97)]
	contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# print('Length of contours-testbefore: right_lablels()',len(contours))
	try:
		if (len(contours) >= 2):
			c = max(contours, key = cv2.contourArea)
			contours.remove(c)
			while (len(contours) >= 2):
				c1 = min(contours, key = cv2.contourArea)
				contours.remove(c1)
	except:
		print('error right label')
	# print('Length of contours-testafter: right_lablels()',len(contours))
	cnt = contours[0]
	S_label = cv2.contourArea(cnt)
	binary = cv2.drawContours(binary, contours, -1, (100,87,51),3)
	hull = [cv2.convexHull(c) for c in contours]
	binary = cv2.drawContours(binary,hull,-1,(100,87,55),3)
	# cv2.imshow('binary label-right_labels',binary)
	label_y_arr = []
	for num, cnt in enumerate(contours):
		x, y, w, h = cv2.boundingRect(cnt)
        # Getting ROI 
		roi = image[y:y+h, x:x+w] 
		# show ROI 
		# cv2.imshow('segment rightlabel no:'+str(num),roi) 
		cv2.rectangle(center_img,(x,y),( x + w, y + h ),(255,0,0),2)
		# cv2.imshow('marked rightlabel areas',center_img) 
		if (y > (BANG_HEIGHT * image_h)):
			label_y_arr.append(y)
			label_y_arr.append(y + h)
	min_label_y, max_label_y = 0, 0
	if label_y_arr:
		min_label_y = min(label_y_arr)
		max_label_y = max(label_y_arr)
	return S_label, min_label_y, max_label_y

# --------------------------------------------------------------------------------------------------------
# Find the liquid level ratio
def find_liquid_level(image):
	image_h, image_w, _ = image.shape
	center_img = image[:, int(image_w * 0.03): int(image_w * 0.97)]
	binary = liqlevel_segmentation(image)
	binary_copy = binary.copy()
	binary_copy = cv2.bitwise_not(binary_copy)
	components = cv2.connectedComponentsWithStats(binary_copy, connectivity=8, ltype=cv2.CV_32S)
	_, labelmap, stats, centers = components
	# print('stats',stats)
	# S of label
	st_s = stats[:, 4]
	s_largest = np.argmax(st_s)
	stats1 = np.delete(stats,s_largest,0)
	# The width of label
	st_w = stats1[:, 2]
	# print('st_w',st_w)
	# y0
	st_y = stats1[:, 1]
	# print('st_y',st_y)
	# print('Test stat1',stats)
	w_largests = [np.argmax(st_w)]
	W = np.where(st_w > (image_w / 2))
	st_w_arg = W[0]
	# print('st argument',st_arg)
	# bin_h,bin_w =bin.shape
	number_label = len(st_w_arg)
	# print('number of label-width comparision(find liquid level):',number_label)
	if (len(st_w_arg) > 1):
		y_min_arg = np.argmin(st_y)
		# print('Largest y index', y_max_arg)
		left_liq = stats1[y_min_arg,0]
		# print('left liquid',left_liq)
		top_liq = stats1[y_min_arg,1]
		# print('top liquid',top_liq)
		# print('top liquid',top_liq)
		right_liq = left_liq + stats1[y_min_arg,2]
		# print('right liquid',left_liq)
		bottom_liq = top_liq + stats1[y_min_arg,3]
	y_min_arg = np.argmin(st_y)
	# print('Largest y index', y_max_arg)
	left_liq = stats1[y_min_arg,0]
	# print('left liquid',left_liq)
	top_liq = stats1[y_min_arg,1]
	# print('top liquid',top_liq)
	# print('top liquid',top_liq)
	right_liq = left_liq + stats1[y_min_arg,2]
	# print('right liquid',left_liq)
	bottom_liq = top_liq + stats1[y_min_arg,3]
	return top_liq

#---------------------------------------------------------------------------------------------------------------
# Find the exact ratio between liquid and bottle
def right_liquid_level():
	image = input_image('D:\\Testimg\\Bottle_checking (CC)\\good1.jpg')
	image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)
	ret, image_binary = cv2.threshold(image_gray,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	image_binary_Gauss = cv2.GaussianBlur(image_binary, (15,15), 15)
	ret2, image_binary2 = cv2.threshold(image_binary_Gauss,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	image_binary2_Gauss = cv2.GaussianBlur(image_binary2, (15,15), 15)
	ret3, image_binary3 = cv2.threshold(image_binary2_Gauss,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# ------------------------------------------------------------------
	# Find and Draw contours
	contours, hierarchy = cv2.findContours(image_binary3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	image_binary3c = cv2.drawContours(image_binary3, contours, -1, (100,87,51),3)
	hull = [cv2.convexHull(c) for c in contours]
	image_binary3c_final = cv2.drawContours(image_binary3,hull,-1,(100,87,55),3)

	#--------------------------------------------------------------------
	# Remove the largest contours
	if (len(contours) >= 3):
		c = max(contours, key = cv2.contourArea)
		contours.remove(c)
		while (len(contours) >= 3):
			c1 = min(contours, key = cv2.contourArea)
			contours.remove(c1)
				
	#-----------------------------------------------
	#Draw ROI with the bottle
	#sort contours
	orig1 =image.copy()
	sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
	# print('Number of contour',len(contours))
	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box 
		x, y, w, h = cv2.boundingRect(ctr) 
    	# Getting ROI 
		roi = image[y:y+h, x:x+w] 
    	# show ROI 
		# cv2.imshow('segment no:'+str(i),roi) 
		# cv2.rectangle(orig1,(x,y),( x + w, y + h ),(255,0,0),2)
		# cv2.imshow('marked areas',orig1) 
	#----------------------------------------------------------------
	#Test draw contour and calculate distance 
	colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255))
	refObj = None
	# loop over the contours individually
	for c in sorted_ctrs:
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 100:
			continue
		# compute the rotated bounding box of the contour
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")	
		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)
		# compute the center of the bounding box
		cX = np.average(box[:, 0])
		cY = np.average(box[:, 1])
		# if this is the first contour we are examining (i.e.,
		# the left-most contour), we presume this is the
		# reference object
		if refObj is None:
			# unpack the ordered bounding box, then compute the
			# midpoint between the top-left and top-right points,
			# followed by the midpoint between the top-right and
			# bottom-right
			(tl, tr, br, bl) = box
			# print('bottom body r', br)
			# print('bottom body l', bl)
			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
			bottle_y2 = int(bl[1])
			bottle_x1 = int(tl[0])
			bottle_x2 = int(tr[1])
			# compute the Euclidean distance between the midpoints,
			# then construct the reference object
			D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
			refObj = (box, (cX, cY), D / 5.78)
			continue
		# draw the contours on the image
		orig = image.copy()
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (50,115,255), -1)
		D1_body = dist.euclidean((tltrX, tltrY), (blbrX, blbrY)) / refObj[2]
		# print('the length of bottle body', D1_body)
		# stack the reference coordinates and the object coordinates
		# to include the object center
		refCoords = np.vstack([refObj[0], refObj[1]])
		objCoords = np.vstack([box, (cX, cY)])
		box_copy = box
		(tl2, tr2, br2, bl2) = box_copy
		# print('top bang r', tr2)
		# print('top bang l', tl2)

		(tltrX2, tltrY2) = midpoint(tl2, tr2)
		(blbrX2, blbrY2) = midpoint(bl2, br2)
		bottle_y1 = int(tl2[1])
		cv2.circle(orig, (int(tltrX2), int(tltrY2)), 5, (50,115,255) , -1)
		cv2.line(orig, (int(blbrX), int(blbrY)), (int(tltrX2), int(tltrY2)),(50,115,255), 2)
		Bottle_height = dist.euclidean((blbrX, blbrY), (tltrX2, tltrY2)) / refObj[2]
		Bottle_position = np.array([bottle_x1,bottle_x2,bottle_y1, bottle_y2])
		# print('the length of bottle', Bottle_height)
		(bottleX, bottleY) = midpoint((blbrX, blbrY), (tltrX2, tltrY2))
		cv2.putText(orig, "{:.1f}cm".format(Bottle_height), (int(bottleX), int(bottleY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,115,255), 2)
		# cv2.imshow('pre Test distance ',orig)
		for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
			# draw circles corresponding to the current points and
			# connect them with a line
			cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
			cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
			cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),color, 2)
			# compute the Euclidean distance between the coordinates,
			# and then convert the distance in pixels to distance in
			# units
			D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
			(mX, mY) = midpoint((xA, yA), (xB, yB))
			cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
		cv2.imshow('Ratio distance ',orig)
	Ratio_liquidvsbottle = D1_body / Bottle_height
	return Ratio_liquidvsbottle, Bottle_position

#-------------------------------------------------------------------------------------------------------------------
def find_bottle(image):

	# Gaussian filter and threshold segmentation (Binary and Otsu)
	image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)
	ret, image_binary = cv2.threshold(image_gray,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	image_binary_Gauss = cv2.GaussianBlur(image_binary, (15,15), 15)
	ret2, image_binary2 = cv2.threshold(image_binary_Gauss,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	image_binary2_Gauss = cv2.GaussianBlur(image_binary2, (15,15), 15)
	ret3, image_binary3 = cv2.threshold(image_binary2_Gauss,50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	binary = image_binary3.copy()
	binary_not = cv2.bitwise_not(binary)
	number_label, mask_bin = ConnectedComponents(binary_not)
	binary = cv2.bitwise_and(binary_not,mask_bin)
	binary = cv2.bitwise_not(binary)
	image_binary4 = binary.copy()

	# Find and Draw contours
	contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# print('len contour-test1',len(contours))	
	# Remove the largest contours
	try:
		if (len(contours) >= 3):
			c = max(contours, key = cv2.contourArea)
			contours.remove(c)
			while (len(contours) >= 3):
				c1 = min(contours, key = cv2.contourArea)
				contours.remove(c1)
	except:
		print('error find bottle')

	image_binary3c = cv2.drawContours(binary, contours, -1, (100,87,51),3)
	hull = [cv2.convexHull(c) for c in contours]
	image_binary3c_final = cv2.drawContours(binary,hull,-1,(100,87,55),3)
	#Draw ROI with the bottle
	#sort contours
	org1 =image.copy()
	sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
	# print('Number of contour',len(contours))
	for i, ctr in enumerate(sorted_ctrs):
		# Get bounding box 
		x, y, w, h = cv2.boundingRect(ctr) 
    	# Getting ROI 
		roi = image[y:y+h, x:x+w] 
    	# show ROI 
		# cv2.imshow('segment no - test:'+str(i),roi) 
		cv2.rectangle(org1,(x,y),( x + w, y + h ),(255,0,0),2)
		cv2.imshow('marked areas - test',org1) 
	
	#Test draw contour and calculate distance 
	colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255))
	refObj = None

	# loop over the contours individually
	# print('length sorted_ctrs',len(sorted_ctrs))
	for c in sorted_ctrs:
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 100:
			continue
		# compute the rotated bounding box of the contour

		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")		
		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)
		# compute the center of the bounding box
		cX = np.average(box[:, 0])
		cY = np.average(box[:, 1])
		# if this is the first contour we are examining (i.e.,
		# the left-most contour), we presume this is the
		# reference object

		if refObj is None:
			# unpack the ordered bounding box, then compute the
			# midpoint between the top-left and top-right points,
			# followed by the midpoint between the top-right and
			# bottom-right
			(tl, tr, br, bl) = box
			(tlblX, tlblY) = midpoint(tl, bl)
			(trbrX, trbrY) = midpoint(tr, br)
			(tltrX, tltrY) = midpoint(tl, tr)
			(blbrX, blbrY) = midpoint(bl, br)
			bottle_y2 = int(bl[1])
			bottle_x1 = int(tl[0])
			bottle_x2 = int(tr[1])

			# compute the Euclidean distance between the midpoints,
			# then construct the reference object
			D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
			D1_body = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
			refObj = (box, (cX, cY), D / 5.72)
			# print('concu1')
			continue
		# draw the contours on the image

		orig = image.copy()
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
		cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (50,115,255), -1)
		# stack the reference coordinates and the object coordinates
		# to include the object center
		refCoords = np.vstack([refObj[0], refObj[1]])
		objCoords = np.vstack([box, (cX, cY)])
		box_copy = box
		(tl2, tr2, br2, bl2) = box_copy
		(tltrX2, tltrY2) = midpoint(tl2, tr2)
		(blbrX2, blbrY2) = midpoint(bl2, br2)
		# print('top of bottle', [tltrX2, tltrY2])
		bottle_y1 = int(tl2[1])
		# print('y bottle test',y_bottle)
		cv2.circle(orig, (int(tltrX2), int(tltrY2)), 5, (50,115,255) , -1)
		cv2.line(orig, (int(blbrX), int(blbrY)), (int(tltrX2), int(tltrY2)),(50,115,255), 2)
		Bottle_height = 0  
		Bottle_height = dist.euclidean((blbrX, blbrY), (tltrX2, tltrY2)) / refObj[2]
		Bottle_position = np.array([bottle_x1,bottle_x2,bottle_y1, bottle_y2])
		(bottleX, bottleY) = midpoint((blbrX, blbrY), (tltrX2, tltrY2))
		cv2.putText(orig, "{:.1f}cm".format(Bottle_height), (int(bottleX), int(bottleY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,115,255), 2)
		# cv2.imshow('pre Test distance ',orig)
		for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
			# draw circles corresponding to the current points and
			# connect them with a line

			cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
			cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
			cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),color, 2)
 
			# compute the Euclidean distance between the coordinates,
			# and then convert the distance in pixels to distance in
			# units
			D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
			(mX, mY) = midpoint((xA, yA), (xB, yB))
			cv2.putText(orig, "{:.1f}cm".format(D), (int(mX), int(mY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
		cv2.imshow('Bottle mask ',orig)
		# cv2.imshow('binary_test2',binary)
	return image_binary4, Bottle_height, D1_body, Bottle_position

# --------------------------------------------------------------------------------------------------------------------
# Draw text result

def draw_textresult(image, label_exist, label_ok, liquid_ok):
	labelstatus = ""
	if (label_exist == False):
		labelstatus = "NO LABEL"
	else:
		if (label_ok == True):
			labelstatus = "OK"
		elif (label_ok == False):
			labelstatus = "BAD"
	cv2.putText(image,"Label position: {:.10s}".format(labelstatus),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,146,152),2)
	# ------------------------------------------------------------
	liquidstatus = ""
	if (liquid_ok == True):
		liquidstatus = "OK"
	elif (liquid_ok == False):
		liquidstatus = "BAD"
	cv2.putText(image,"Liquid volume: {:.10s}".format(liquidstatus),(10,40), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,146,152),2)
	return image

# -----------------------------------------------------------------------------------------------------------------------
# Handle image to process image

def handle_image(img,rightlabel,rightliquid):

	right_liquid_max = rightliquid[0]
	right_liquid_min = rightliquid[1]

	S_good_label = rightlabel[0]
	label_rightbottom = rightlabel[1]
	label_rightheight = rightlabel[2]
	label_rightbottom_max = rightlabel[3]
	label_rightbottom_min = rightlabel[4]
	label_rightheight_max = rightlabel[5]
	label_rightheight_min = rightlabel[6]
	# img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
	img_h, img_w,  _ = img.shape
	img_e = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
	border = int(img_h * CUT_BORDER)
	image = img[:, border:(img_w -border)]
	image_h, image_w, _ = image.shape
	image_original = np.copy(image)

	# pre_roi = get_roi(image, int(image_w * PRE_ROI_WIDTH), int(image_w - image_w * PRE_ROI_WIDTH))
	pre_roi = get_roi(image, int(image_w * 0.2), int(image_w * 0.8))
	pre_mask = cv2.bitwise_not(find_background(image, pre_roi))

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	pre_mask = cv2.morphologyEx(pre_mask, cv2.MORPH_OPEN, kernel)

	bin1, Bottle_height,Bottle_body,Bottle_position = find_bottle(image)
	left_border, right_border, roi_width = get_pre_borders (bin1)
	cut_img1 = image[:, (left_border - roi_width):(right_border + roi_width)]
	cut_img1_h, cut_img1_w, _ = cut_img1.shape
	bottle_test1 = cut_img1.copy()
	bottle_test2 = cut_img1.copy()
	bottle_test3 = cut_img1.copy()

	# cv2.imshow('cut Image1', cut_img1)
	roi = get_roi(cut_img1, roi_width, cut_img1_w - roi_width)
	ROI_EXT = int(cut_img1_w * 0.05)
	cut_img2 = cut_img1[:, (roi_width - ROI_EXT):(cut_img1_w - roi_width + ROI_EXT)]
	cut_img2_h, cut_img2_w, _ = cut_img2.shape
	ROI_EXT21 = int(cut_img2_w * 0.05)
	ROI_EXT22 = int(cut_img2_h * 0.05)
	# cv2.imshow('cut Image2', cut_img2)
	cut_bin,Bottle_height1, _,Bottle_position1 = find_bottle(cut_img2)
	# print('Bottle positon 1',Bottle_position1)
	# Get bottle by mask
	cut_img = cut_img2[(Bottle_position1[2] - ROI_EXT22):(Bottle_position1[3]+ROI_EXT22) , (Bottle_position1[0]-ROI_EXT21):(Bottle_position1[1]+ROI_EXT21)]
	bottle = cut_img
	bottle_h, bottle_w, _ = bottle.shape
	cv2.imshow('Bottle', bottle)
	BottleHeight = (Bottle_position1[3] - Bottle_position[2])

	# -------------------------------------------------------------------------
	# Check the quality of label
	# -------------------------------------------------------------------------
	
	labelexist = check_label_exist(bottle_test1,S_good_label)

	# Check the existence of label
	if (labelexist == True):
		print('Bottle has label')
	else:
		print('Bottle has no label')
	label_ok = False
	label_bottom = 0
	label_height = 0
	# Check the position and the height of label
	if (labelexist == True):
		minlabely,maxlabely = find_labels(bottle_test2)
		# print('min label y', minlabely)
		# print('max label y', maxlabely)
		label_bottom = 1 - ((maxlabely - Bottle_position1[2]) / (Bottle_position1[3] - Bottle_position[2]))
		label_height = (maxlabely - minlabely) / (Bottle_position1[3] - Bottle_position1[2])
		print('label_bottom',label_bottom)
		print('label_height',label_height)
			
		if (label_bottom > label_rightbottom_min) and (label_bottom < label_rightbottom_max):
			if (label_height > label_rightheight_min) and (label_height < label_rightheight_max):
				label_ok = True
		print('Check if label is ok: ', label_ok)

	# -------------------------------------------------------------------------
	# Check the liquid level
	# -------------------------------------------------------------------------
		
	liquid_level_y = find_liquid_level(bottle_test3)
	liquid_level = 1 - ((liquid_level_y - Bottle_position1[2]) / (Bottle_position1[3] - Bottle_position[2]))
	# print('top of bottle',Bottle_position1[2])
	# print('bottom of bottle',Bottle_position[3])
	print('liquid level', liquid_level)
	liquid_ok = False
	if (liquid_level > right_liquid_min) and (liquid_level < right_liquid_max):
		liquid_ok = True
	print('Check if liquid level is ok: ', liquid_ok)
		
	recognize_info = label_height, label_bottom, liquid_level,labelexist,label_ok,liquid_ok

	# ------------------------------------------------------------------------
	# Draw text result to image
	# -------------------------------------------------------------------------

	result_image = draw_textresult(image,labelexist,label_ok,liquid_ok)
	cv2.imshow('the result of bottle checking', image)
	return result_image, recognize_info

# ----------------------------------------------------------------------------------------------------------------------
# Save image
# def save_img(result_image, result_path, file_name):

#------------------------------------------------------------------------------------------------------------------------
# Capture camera from webcam

def capture_cam(captures_path, result_path,right_label_matrix, right_liquid_matrix):
	cap = cv2.VideoCapture(0)
	cap.set(3, 1280)
	cap.set(4, 960)
	recognize_info = None
	while (True):
		ret, frame = cap.read()
		
		if not ret:
			print("No capture")
			break
		# recognize_info = label_height, label_bottom, liquid_level,labelexist,label_ok,liquid_ok
		if recognize_info:
			# frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
			frame = draw_textresult(frame, recognize_info[3],recognize_info[4],recognize_info[5])
			# frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		cv2.imshow('Bottle check', frame)
		k = cv2.waitKey(100)
		if k == 27:  # Esc key to stop
			break
		elif k == 32:  # Space key to recognize
			# Upon the initial click of <Space> the current frame is
			# recognized, the rendering of the results in the display
			# window received from the webcam and saving of the images
			# are performed.
			# The second click on <Space> will clear the rendering of the
			# results of recognition and make the process ready to repeat itself.
			if recognize_info:
				recognize_info = None
			else:
				file_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f.png")
				cv2.imwrite(os.path.join(captures_path, file_name), frame)
				result_image, recognize_info = handle_image(frame,right_label_matrix, right_liquid_matrix)
				# save_img(result_image, result_path, file_name)
				cv2.imwrite(os.path.join(result_path, file_name),result_image)
	cap.release()
	cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------------
# Take video from path and process

def capture_video(video_path, captures_path, result_path,right_label_matrix, right_liquid_matrix):
	cap = cv2.VideoCapture(video_path)
	cap.set(3, 1280)
	cap.set(4, 960)
	recognize_info = None
	while (True):
		ret, frame = cap.read()

		if not ret:
			print("No capture")
			break
		if recognize_info:
			# frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
			frame = draw_textresult(frame,  recognize_info[3],recognize_info[4],recognize_info[5])
			# frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		cv2.imshow('Bottle check', frame)
		k = cv2.waitKey(100)
		if k == 27:  # Esc key to stop
			break
		elif k == 32:  # Space key to recognize
			# Upon the initial click of <Space> the current frame is
			# recognized, the rendering of the results in the display
			# window received from the webcam and saving of the images
			# are performed.
			# The second click on <Space> will clear the rendering of the
			# results of recognition and make the process ready to repeat itself.
			if recognize_info:
				recognize_info = None
			else:
				file_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f.png")
				cv2.imwrite(os.path.join(captures_path, file_name), frame)
				# frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
				result_image, recognize_info = handle_image(frame,right_label_matrix, right_liquid_matrix)
				# save_img(result_image, result_path, file_name)
				cv2.imwrite(os.path.join(result_path, file_name),result_image)
	cap.release()
	cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------------------------------

def main():
		
		# --------------------------------------------------------------
		# Find the exact ratio between liquid level and the height of bottle
		# Find the exact ratio between the height of label and the height of bottle
		#---------------------------------------------------------------
		right_ratio, Bottle_rightposition = right_liquid_level()
		right_liquid_max = right_ratio + 0.02
		right_liquid_min = right_ratio - 0.02
		right_liquid_matrix = right_liquid_max, right_liquid_min
		print('Ratio between the height of bottle and the liquid level', right_ratio)
		
		S_good_label,min_rightlabel_y, max_rightlabel_y = right_labels()
		S_good_label = int(S_good_label)
		# print('The area of good label', S_good_label)
		label_rightbottom = 1 - ((max_rightlabel_y - Bottle_rightposition[2]) / (Bottle_rightposition[3] - Bottle_rightposition[2]))
		label_rightheight = (max_rightlabel_y - min_rightlabel_y) / (Bottle_rightposition[3] - Bottle_rightposition[2])
		label_rightbottom_max = label_rightbottom + 0.032
		label_rightbottom_min = label_rightbottom - 0.032
		label_rightheight_max = label_rightheight + 0.01
		label_rightheight_min = label_rightheight - 0.01
		right_label_matrix = S_good_label, label_rightbottom, label_rightheight, label_rightbottom_max, label_rightbottom_min, label_rightheight_max, label_rightheight_min
		print('label_rightbottom_max ',label_rightbottom_max )
		print('label_rightbottom_min',label_rightbottom_min)
		print('label_rightheight_max',label_rightheight_max)
		print('label_rightheight_min',label_rightheight_min)

		# # -------------------------------------------------------------------------
		# # Find bottle
		# # -------------------------------------------------------------------------
		# # Read image from directory 
		# img = input_image('D:\\Testimg\\Bottle_checking (CC)\\lowliq24.jpg')
		# img_h, img_w,  _ = img.shape
		# img_e = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		# border = int(img_h * CUT_BORDER)
		# image = img[:, border:(img_w -border)]
		# image_h, image_w, _ = image.shape
		# image_original = np.copy(image)

		# # pre_roi = get_roi(image, int(image_w * PRE_ROI_WIDTH), int(image_w - image_w * PRE_ROI_WIDTH))
		# pre_roi = get_roi(image, int(image_w * 0.2), int(image_w * 0.8))
		# pre_mask = cv2.bitwise_not(find_background(image, pre_roi))

		# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
		# pre_mask = cv2.morphologyEx(pre_mask, cv2.MORPH_OPEN, kernel)

		# bin1, Bottle_height,Bottle_body,Bottle_position = find_bottle(image)
		# left_border, right_border, roi_width = get_pre_borders (bin1)
		# cut_img1 = image[:, (left_border - roi_width):(right_border + roi_width)]
		# cut_img1_h, cut_img1_w, _ = cut_img1.shape
		# bottle_test1 = cut_img1.copy()
		# bottle_test2 = cut_img1.copy()
		# bottle_test3 = cut_img1.copy()

		# # cv2.imshow('cut Image1', cut_img1)
		# roi = get_roi(cut_img1, roi_width, cut_img1_w - roi_width)
		# ROI_EXT = int(cut_img1_w * 0.05)
		# cut_img2 = cut_img1[:, (roi_width - ROI_EXT):(cut_img1_w - roi_width + ROI_EXT)]
		# cut_img2_h, cut_img2_w, _ = cut_img2.shape
		# ROI_EXT21 = int(cut_img2_w * 0.05)
		# ROI_EXT22 = int(cut_img2_h * 0.05)
		# # cv2.imshow('cut Image2', cut_img2)
		# cut_bin,Bottle_height1, _,Bottle_position1 = find_bottle(cut_img2)
		# # print('Bottle positon 1',Bottle_position1)
		# # Get bottle by mask
		# cut_img = cut_img2[(Bottle_position1[2] - ROI_EXT22):(Bottle_position1[3]+ROI_EXT22) , (Bottle_position1[0]-ROI_EXT21):(Bottle_position1[1]+ROI_EXT21)]
		# bottle = cut_img
		# bottle_h, bottle_w, _ = bottle.shape
		# cv2.imshow('Bottle', bottle)
		# BottleHeight = (Bottle_position1[3] - Bottle_position[2])
		# # -------------------------------------------------------------------------
		# # Check the quality of label
		# # -------------------------------------------------------------------------
	
		# labelexist = check_label_exist(bottle_test1,S_good_label)

		# # Check the existence of label
		# if (labelexist == True):
		# 	print('Bottle has label')
		# else:
		# 	print('Bottle has no label')
		# label_ok = False
		# # Check the position and the height of label
		# if (labelexist == True):
		# 	minlabely,maxlabely = find_labels(bottle_test2)
		# 	# print('min label y', minlabely)
		# 	# print('max label y', maxlabely)
		# 	label_bottom = 1 - ((maxlabely - Bottle_position1[2]) / (Bottle_position1[3] - Bottle_position[2]))
		# 	label_height = (maxlabely - minlabely) / (Bottle_position1[3] - Bottle_position1[2])
		# 	print('label_bottom',label_bottom)
		# 	print('label_height',label_height)
			
		# 	if (label_bottom > label_rightbottom_min) and (label_bottom < label_rightbottom_max):
		# 		if (label_height > label_rightheight_min) and (label_height < label_rightheight_max):
		# 			label_ok = True
		# 	print('Check if label is ok: ', label_ok)

		# # -------------------------------------------------------------------------
		# # Check the liquid level
		# # -------------------------------------------------------------------------
		
		# liquid_level_y = find_liquid_level(bottle_test3)
		# liquid_level = 1 - ((liquid_level_y - Bottle_position1[2]) / (Bottle_position1[3] - Bottle_position[2]))
		# # print('top of bottle',Bottle_position1[2])
		# # print('bottom of bottle',Bottle_position[3])
		# print('liquid level', liquid_level)
		# liquid_ok = False
		# if (liquid_level > right_liquid_min) and (liquid_level < right_liquid_max):
		# 	liquid_ok = True
		# print('Check if liquid level is ok: ', liquid_ok)
		
		# recognize_info = label_height, label_bottom, label_height,liquid_level,labelexist,label_ok,liquid_ok

		# # ------------------------------------------------------------------------
		# # Draw text result to image
		# # -------------------------------------------------------------------------

		# draw_textresult(image,labelexist,label_ok,liquid_ok)
		# cv2.imshow('the result of bottle checking', image)
		# ---------------------------------------------------------------------------
		# Test
		# ----------------------------------------------------------------------------
		# img = input_image('D:\\Testimg\\Bottle_checking (CC)\\good4.jpg')
		# resultimg, recognize_info = handle_image(img,right_label_matrix, right_liquid_matrix)
		# print('recognize_info', recognize_info)
		capturepath = "D:\\Testimg\\Capture"
		resultpath = "D:\\Testimg\\Result"
		videopath = "D:\\Testimg\\Bottle_video(CC)\\video_test7.mp4"
		capture_video(videopath,capturepath,resultpath,right_label_matrix, right_liquid_matrix)
		# while True:
			
		# 	key = cv2.waitKey(30)
		# 	if key == ord('q') or key == 27:
		# 		break

if __name__ == "__main__":
    main()



