# -*- coding: utf-8 -*- 
from tkinter import *
import tkinter.scrolledtext as tkst
from tkinter import ttk
from tkinter.filedialog import (askopenfilename, 
                                askopenfilenames, 
                                askdirectory, 
                                asksaveasfilename)
from PIL import Image, ImageTk
import cv2
import numpy as np
import math,random,os
# fcn
from FCN_DatasetReader import DatasetReader, ImageReader
import scipy.misc as misc
from scipy.signal import argrelextrema
import tensorflow as tf
from shutil import copyfile
import FCN_model,FCN_CrackAnalysis
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# WEIGHTS = np.load('vgg19_weights.npy', encoding='bytes').item()

# ************** Original ********************** #
SHOW_PIXELS_HISTOGRAM = False

def least_square(img):
	pos1 = np.where(img)
	y = pos1[0]
	x = pos1[1]
	pos_num = len(x)
	one = np.ones((pos_num,1))
	X = np.c_[x,one]
	XX = np.linalg.inv(X.T.dot(X))
	w_ = XX.dot(X.T.dot(y))
	return w_

def rotate_image(img,theata,color=False,bV=(0,0,0)):
	if color:
		rows,cols,channels = img.shape
	else:
		rows, cols = img.shape
	# 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
	M = cv2.getRotationMatrix2D((0,0),theata,1) # 旋转参考点
	dst = cv2.warpAffine(img,M,(cols,rows),borderValue=bV)
	return dst

'''
把np.where得到的坐标转换成cnt轮廓的格式
输入：np.where得到的tuple
输出：cnt格式的array
'''
def where2contour(itup):
	ytup,xtup = itup
	num = len(xtup)
	resarr = np.zeros(shape=(num,1,2),dtype=np.int32)
	for i in range(0,num):
		p = np.array([xtup[i],ytup[i]]).reshape(1,2)
		resarr[i] = p
	return resarr

# *********************** Original End **************** #
# *********************** FCN ************************** #
def save_png(file_name, ndarray, new_size):
    image = np.squeeze(ndarray)
    new_image = misc.imresize(image, size=new_size)
    misc.imsave(file_name, new_image)

# *******************************************************#

def findCrack(fileName, output='./demo.jpg'):
	img = cv2.imread(fileName)

	# 第一步：从图像中裁剪出手机的区域。ostu二值化，然后阈值，算极点，剪切一个矩形
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,th1 = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV) # 阈值是一个需要调节的参数
	edges01 = cv2.Canny(gray,100,196) # 找出手机区域裁剪
	blur = cv2.GaussianBlur(th1,(5,5),0)
	# 旋转水平,最小旋转矩形框第一步canny的结果
	pos = np.where(edges01)
	cnt_p = where2contour(pos)
	rect = cv2.minAreaRect(cnt_p)
	# box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
	# box = np.int0(box)
	# cv2.drawContours(img,[box],0,(0,0,255),2)
	# 测试完成，矩形框出了手机，用矩形的角度旋转图片. 这里旋转角度可能先碰到较短的边
	h1,h2 = rect[1]
	theata01 = rect[2] if h1>h2 else 90+rect[2]

	rotated_b = rotate_image(blur,theata01)
	rotated_i = rotate_image(img,theata01,True,(255,255,255))
	rotated_g = rotate_image(gray,theata01)
	rotated_e = rotate_image(edges01,theata01)

	# 求极点
	pos=np.where(rotated_e)
	leftmost = min(pos[0])
	rightmost = max(pos[0])
	topmost = max(pos[1])
	bottommost = min(pos[1])

	range1 = rotated_i[leftmost:rightmost, bottommost:topmost]
	range1g = rotated_g[leftmost:rightmost, bottommost:topmost]
	range_blur = rotated_b[leftmost:rightmost, bottommost:topmost]
	
	# 第三步：投影法检测到缝隙的位置。用直线近似拟合表示
	print('>>> 3.定位缝隙，测量缝宽')



	# 直方图均衡化，增强图像的对比度,更容易检测缝隙？
	equ = cv2.equalizeHist(range1g)
	equ_show = cv2.cvtColor(equ,cv2.COLOR_GRAY2BGR)
	# *噪声以孤立点的形式存在，采用中值滤波可以降噪滤波，去除图像中高频噪点，裂缝信息可以得到保护
	# 试一下中值滤波; 第二个参数卷积核大小，应该是个奇数
	med = cv2.medianBlur(equ, 9)
	# 直方图均衡化之后再全局阈值，把缝隙留下
	# equ_ret,equ_th = cv2.threshold(equ, 180, 255, cv2.THRESH_BINARY)
	# 自适应阈值调节，光照不一致，图片不同部分有不同亮度

	edges = cv2.Canny(med, 50, 100) # minVal和maxVal是需要调节的阈值
	# edges0 = cv2.Canny(th3,100,200)
	# w_ = least_square(edges0)

	# # 将图像旋转至水平
	rows, cols = th1.shape


	dst_s = range1

	'''
	在最终结果上画线
	'''
	def draw_line(line_set,c=(0,0,255)):
		for l in line_set:
			# print('line:', l,'Intensity: ',gray_r[l])
			min1_f = l
			# cv2.line(dst_s,(min(l1),min1_f),(max(l1),min1_f),(0,0,255),1)
			cv2.line(dst_s,(0,min1_f),(range1.shape[1],min1_f),c,1)
			# cv2.line(equ_show,(min(l1),min1_f),(max(l1),min1_f),(0,0,255),1)

	# 投影法，累计水平像素灰度值,边缘图or灰度图
	gray_r = np.sum(equ,axis=1)
	range1g_r = np.sum(range1g,axis=1)
	# 找到直方图中的局部极小值点，这一个是一个非常好的预选
	xrelative_extrema = argrelextrema(range1g_r,np.less) # np.less_equal
	print('投影直方图局部极小值点:',xrelative_extrema,type(xrelative_extrema))
	xree_intensity = range1g_r[xrelative_extrema]
	# print('xree_intensity:',xree_intensity)
	xrelative_extrema0 = xrelative_extrema[0]

	# 缝隙的灰度值很小，在整个图片来说;在其领域附近来说
	gmean = int(np.mean(range1g_r))
	inten_thr = int(0.9*gmean) # 灰度值要小于阈值，需要设定
	xree_aindex = np.where(xree_intensity < inten_thr)
	# print(type(xree_amean),xree_amean)
	xree_amean = xrelative_extrema0[xree_aindex]
	print('xree_amean:',xree_amean,type(xree_amean),'<<<红线')
	# draw_line(xree_amean,(0,0,255))

	# 注意到手机上表面和下表面都有灰度变化明显的特点。
	# 投影灰度值最小的位置,在xree_mean中选灰度最小的
	xree_amean_inten = range1g_r[xree_amean]
	print('在均值以下的局部极小值对应的灰度值：',xree_amean_inten)
	print('最右边的局部极小值点位置(被去掉）：',xree_amean[-1])
	# print('xree_amean_inten',xree_amean_inten)
	g_sort = xree_amean_inten.argsort()
	if g_sort[0] == len(xree_amean)-1:
		x_pos = xree_amean[g_sort[1]] # 去掉最右边的局部极小值，可能是桌面阴影
	else:
		x_pos = xree_amean[g_sort[0]]
	draw_line((x_pos,),(0,255,0))
	print('灰度最小的局部极小值点（去掉最右边点）：',x_pos,'<<<绿线')

	'''
	输入：灰度投影的np.array数组
	输出：差分的np.array数组
	'''
	def getArrDiff(dim1arr):
		b = np.zeros(len(dim1arr))
		b[0:-1] = dim1arr[1:]
		diff = b - dim1arr
		diff[-1] = 0
		return diff

	'''
	差分数组中最大值，位置和其对应（最相近）的局部极小值点。
	因为最左边和最右边边界问题，所以要从差分数组从大到小查找，直到不是边界，返回这个局部极小值点
	输入：投影图差分绝对值数值
	输出：梯度最大位置对应的局部极小值点位置（去掉最右边点）,差分位置,差分值
	'''
	def getPnotEdge(diffar,xram):
		res = 0
		idarr = np.argsort(diffar)
		for i in range(len(idarr)-1,-1,-1):
			ide = idarr[i]
			# 数组中和给定数最接近的数
			idx = abs(xram - ide).argmin()
			if xram[idx] != xram[-1]: 
				res = ( xram[idx], ide, diffar[ide] )
				break
		return res


	# 对灰度图求差分
	diff_range1gr = getArrDiff(range1g_r)
	diff_range1gr_abs = abs(diff_range1gr)
	diff_pos_max = np.argmax(diff_range1gr)
	# print('投影图差分：',diff_range1gr)
	diff_pos_max_abs = np.argmax(diff_range1gr_abs)
	print('投影图差分绝对值最大值位置：',diff_pos_max_abs)
	ree_diff_max = getPnotEdge(diff_range1gr_abs,xree_amean)
	print('梯度最大位置邻近的局部极小值点（去掉最右边点）,差分位置，差分值：',ree_diff_max,'<<<蓝线')
	draw_line((ree_diff_max[0],),(255,0,0))

	# 二阶差分
	diff2 = getArrDiff(diff_range1gr)

	# 在这一条线上做文章，找缝隙位置：掩膜相关运算；找缝隙大小，这条线上的边缘检测
	# 参数 1,0 为只在 x 方向求一阶导数，最大可以求 2 阶导数。 
	# sobelx=cv2.Sobel(range1g,cv2.CV_64F,1,0,ksize=5)

	# 1.找缝隙位置，一维高斯掩膜相关运算

	'''
	计算投影图中的V区域
	输入：灰度投影数组，左右边界和局部极小值点（left，rela_min，right)求V区域边界
	输出：三元组
	阈值需要设定
	思想：左右两边都是一个斜率较大的部分。
	应该：合缝应该是直到遇到一个剧烈变化即斜率较大的地方停止。<<<TODO
	'''
	def getVarea(gimgh,lpr,diff_thr=0.1):
		lefte,p,righte = lpr
		l,r = lefte,righte
		# 右边
		suM, num, mean = 0,0,0.0
		for pi in range(p,righte):
			diff = gimgh[pi+1] - gimgh[pi]
			if diff < mean*diff_thr:
				r = pi
				break
			suM += diff
			num += 1
			mean = suM/num
		# 左边同理
		suM, num, mean = 0,0,0.0
		for pj in range(p,lefte,-1):
			diff = int(gimgh[pj]) - int(gimgh[pj-1])
			if diff > mean * diff_thr:
				l = pj
				break
			suM += diff
			num += 1
			mean = suM/num
		return (l,p,r)

	'''
	求V区域的边界,根据局部极小值点两边的局部极大值点
	'''
	def getVBorder(gimgh):
		center = xree_amean
		xree_emax = argrelextrema(gimgh,np.greater)[0]
		rs=[]
		for p in center:
			# 左右的极大值点作为预选边界点
			tp = (xree_emax - p)
			tp[tp<0] = 600
			righte = xree_emax[tp.argmin()]
			tp = (p - xree_emax)
			tp[tp<0] = 600
			lefte = xree_emax[tp.argmin()]
			if righte < p: righte = p
			if lefte > p: lefte = p
			rs.append((lefte,p,righte))
		return rs

	'''
	检测投影直方图中的‘深V’区域
	'''
	def detectAllVarea(gimgh):
		# 所有局部极小值点预选；深V所以去掉灰度值比较大的点，这里设定(阈值为均值*0.9
		cand = xree_amean
		# 点左边的斜率一直是负的，右边一直是正的。而且斜率的绝对值都要比较大
		# 怎么认为斜率比较大是一个判定，或者阈值
		xree_emax = argrelextrema(gimgh,np.greater)[0]
		vboard = getVBorder(gimgh)
		rs = []
		for lpr in vboard:
			rs.append(getVarea(gimgh,lpr))
		return rs

	Va = detectAllVarea(range1g_r)
	print('基于灰度值极小值点（均值阈值以下）的V区域：',Va)


	print('\t Canny算法提取合缝')
	'''
	判断手机是黑色还是白色
	返回：黑色0，白色1
	'''
	def whiteorBlack(img):
		# 统计灰度直方图，峰值和0和255哪个更接近
		hist = cv2.calcHist([img],[0],None,[256],[0,256])
		# print('hist:',hist,type(hist),len(hist))
		midx = hist.argmax()
		# print(midx)
		ans = 0 if (255-midx) > midx else 1
		# ans = 0 if midx < 70 else 1
		return ans
		
	p_c = whiteorBlack(range1g)
	print('手机颜色：',p_c)
	# 把缝隙的区域提取出来,Canny,配合上一步的预选位置
	hminV = [f for f in Va if f[1]==x_pos][0]
	diffmaxV = [f for f in Va if f[1]==ree_diff_max[0]][0]
	print('灰度值最小V2区域，梯度最小V2区域',hminV,diffmaxV,'<<<绿框，蓝框')
	r,c = range1g.shape
	# cv2.rectangle(dst_s,(0,hminV[0]),(c,hminV[2]),(0,255,0),1)
	# cv2.rectangle(dst_s,(0,diffmaxV[0]),(c,diffmaxV[2]),(255,0,0),1)

	edges1 = cv2.Canny(equ, 110, 210) # minVal和maxVal是需要调节的阈值
	edges3 = cv2.Canny(range1g,100,190)

	width3 = (hminV[2] - hminV[0]) if p_c == 0 else (diffmaxV[2] - diffmaxV[0])
	print('>>>测量的缝宽：根据投影法V区域',width3)

	# 根据投影法得到的预选区域，配合Canny法进行检测
	# 把绿线和蓝线画到 Canny检测结果上看一下位置
	cv2.line(edges1,(0,hminV[1]),(c,hminV[1]),(255,255,255),1)
	cv2.line(edges3,(0,diffmaxV[1]),(c,diffmaxV[1]),(255,255,255),1)
	# 在lpr的预选区域内求解
	vboard = getVBorder(range1g_r)
	hminVBd = [v for v in vboard if v[1]== x_pos][0]
	diffmaxVBd = [v for v in vboard if v[1]==ree_diff_max[0]][0]

	
	# 黑色手机因为Canny效果不好所以不好，裁剪出预选的合缝区域做Canny看看
	blackroi = range1g[hminVBd[0]:hminVBd[2],:]
	# 局部处理；直接canny不行
	# 灰度变化试一下，3个方法：gamma变换，log变换，指数图像增强：scikit iamge
	br_edges = cv2.Canny(blackroi,30,120)
	br_equ = cv2.equalizeHist(blackroi)
	br_equ_edges = cv2.Canny(br_equ,110,490)
	br_blur = cv2.GaussianBlur(br_equ,(5,5),0)
	br_med = cv2.medianBlur(br_equ,5)
	br_med_edges = cv2.Canny(br_med,100,200)
	# print('br_med_edges',br_equ_edges[br_equ_edges>0])
	
	cntup = []
	cntdown = []
	anno_bimg = edges1.copy()
	anno_bimg[:,:] = 0
	cnt_edges = edges1.copy()
	cnt_edges[:,:] = 0
	if p_c == 1:
		edge_use = edges3
		cc = diffmaxV[1]
		board = diffmaxV#
	elif p_c == 0:
		edge_use = edges1
		# 用局部处理的结果代替
		edge_use[hminVBd[0]:hminVBd[2],:] = br_med_edges
		cc = hminV[1]
		board = hminV#
	print('上下边界',board[0],board[2])
	for pi in range(0,c):
		col = edge_use[:,pi]
		colw = np.where(col)[0]
		if len(colw) == 0: 
			cntup.append((cc,pi))
			cntdown.append((cc,pi))
			continue
		# print(colw)
		# 线的上下边界
		tp = (colw - cc)
		# print('tp',tp)
		tp[tp<=0] = 600
		righte = colw[tp.argmin()]
		tp = (cc - colw)
		tp[tp<=0] = 600
		lefte = colw[tp.argmin()]
		if righte < cc: righte = cc
		if lefte > cc: lefte = cc
		# 边界超过board就舍掉
		if righte > board[2]: righte = cc
		if lefte < board[0]: lefte = cc
		# print('上，中心,下',cc,righte,lefte)
		anno_bimg[cc,pi] = 255
		anno_bimg[lefte:righte,pi] = 255
		cnt_edges[lefte,pi] = 255
		cnt_edges[righte,pi] = 255

		cntup.append((lefte,pi))
		cntdown.append((righte,pi))


	# 把找到的边界画到图片上看看
	for p in cntup:
		# dst_s[p[0],p[1]] = (0,0,255)
		cv2.circle(dst_s,(p[1],p[0]),1,(0,0,255),-1)
	for p in cntdown:
		# dst_s[p[0],p[1]] = (0,0,255)
		cv2.circle(dst_s,(p[1],p[0]),1,(0,0,255),-1)

	
	# 将检测到的边界转换成一个区域，然后用骨架提取
	cv2.imwrite('./tmp/anno_bimg.png',anno_bimg)

	analyser = FCN_CrackAnalysis.CrackAnalyse('./tmp/anno_bimg.png')
	crack_max_width = analyser.get_crack_max_width()
	print('骨架提取合缝宽度',crack_max_width)
	cv2.imwrite('./tmp/dip.jpg',dst_s)
	
	return '%.1f'%crack_max_width


'''
从标注的二值图像中计算合缝最大宽度
输入：二值图像,方法参数
输出：宽度值
'''
def getGapWidth(binImg_n,method='skeleton'):
	# 两种方法的评测
	if method == 'minAreaRect':
		binImg = cv2.imread(binImg_n,0)
		# ret,thresh = cv2.threshold(binImg,127,255,cv2.THRESH_BINARY)
		# print(thresh)
		image, contours, hierarchy = cv2.findContours(binImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		contours.sort(reverse=True, key = lambda e:cv2.contourArea(e))
		cnt = contours[0]
		print('getGapWidth --> len(contours):',len(contours))
		rect = cv2.minAreaRect(cnt)
		return min(rect[1])
	elif method == 'skeleton':
		# 计算的时间不短
		analyser = FCN_CrackAnalysis.CrackAnalyse(binImg_n)
		# crack_skeleton = analyser.get_skeleton()
		# crack_length = analyser.get_crack_length()
		crack_max_width = analyser.get_crack_max_width()
		# crack_mean_width = analyser.get_crack_mean_width()
		return '%.1f' % crack_max_width

'''
200张数据集标注的缝宽
'''

def getData200Width(outf='./tmp/gapwidthMedmph.txt'):
	# path = './test/'
	path='../data200/annotation/'
	fl = os.listdir(path)
	fl = [f for f in fl if f.startswith('0')]
	fl = [f for f in fl if f.endswith('png')]
	fl.remove('0010.png')
	fl.remove('0011.png')
	fl.remove('0012.png')
	fl.remove('0016.png')


	fl.sort()
	file = open(outf,'w')
	for f in fl:
		print(f)
		gw = getGapWidth(path+f,'skeleton')
		file.write(f+'\t'+str(gw)+'\n')
	file.close()



def init_fcn():
	pass


def findCrack_fcn(fname='./tmp/demo.jpg',output='./tmp/fcn.png'):
	print(">>> Loading images from test directory ...")
	# inum = 1
	# if isinstance(fname,tuple):
	# 	inum = len(fname)
	# 	image_all = []
	# 	for fni in fname:
	# 		image = misc.imread(fni,mode='RGB')
	# 		if len(image.shape) == 2:
	# 			image = np.expand_dims(image, axis=2)
	# 			resized_image = misc.imresize(image,size=[128,1024])
	# 			image_all.append(resized_image)
	# 	image_all = np.array(image_all)
	# else:
	image = misc.imread(fname, mode='RGB')
	if len(image.shape) == 2:
		image = np.expand_dims(image, axis=2)


	if tf.train.latest_checkpoint('./checkpoints/'):
	    print("Load model from {}".format(tf.train.latest_checkpoint('./checkpoints/')))
	else:
	    print("Please train model first !!!")

    # predict - process
	print("============>>>> Begin to predict ... <<<<============")
	with tf.Session(config=config) as sess:
		saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
		start_time = time.time()

		# image, save_name, save_shape = image_set.next_image()

		resized_image = misc.imresize(image, size=[128, 1024])

		# resized_image = np.expand_dims(image_all, axis=0)
		# resized_image = image_all
		resized_image = np.expand_dims(resized_image, axis=0)

		prediction = sess.run(predictions, feed_dict={img_holder: resized_image})
		# print(type(prediction),prediction.shape)
		# w_all = []
		# for i in inum:
		# 	sni = './tmp/%s'% fname[i].split('/')[-1].replace('jpg','png')
		# 	save_png(sni,prediction[i,:,:,:])
		# 	w = getGapWidth(sni)
		# 	w_all.append(w)
		# return tuple(w_all)
		save_png(output, prediction, image.shape)

		predict_time = time.time() - start_time
		print('Time: %.3f sec' % (predict_time))

		print("============>>>> Finish predict ... <<<<============")
		w=getGapWidth(output)
		return w

if __name__=='__main__':
	print(">>> Setting up FCN model ...")

	# model - input
	img_holder, ant_holder = FCN_model.input(128, 1024)

	layer_name=['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3',
				'conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
	weights_tmp = {n:(None,None) for n in layer_name}

	# model - inference
	logits, predictions = FCN_model.inference(img_holder, 2, weights_tmp, 0.5)

	# model - loss
	loss_op = FCN_model.loss(logits, ant_holder)
	loss_op_ = FCN_model.loss(logits, ant_holder)

	# model - evaluate
	accuracy = FCN_model.evaluate(predictions, ant_holder)
	# precision, recall, f_score, matthews_cc = FCN_model.statistics(predictions, ant_holder)

	# model - train var list
	var_list = tf.trainable_variables()

	# model - train
	train_op = FCN_model.train(loss_op, 1e-4, var_list)

	print(">>> Setting up FCN writer and saver ...")
	saver = tf.train.Saver()

	# 选择图片之后清空label的输出
	def clear():
		sLab_dip_crack.image=None
		sLab_fcn_crack.image=None
		text.delete('1.0',END)

	def hidesingleorshow(flag):
		if flag:
			sLab.grid_forget()
			sLab_dip_crack.grid_forget()
			sLab_fcn_crack.grid_forget()
			text['height']=13
		else:
			sLab.grid(row=1, column=0,columnspan=5)
			sLab_fcn_crack.grid(row=3,column=0,columnspan=5)
			sLab_dip_crack.grid(row=2,column=0,columnspan=5)
			text['height']=5


	def choosepic():
		clear()
		hidesingleorshow(False)
		path_=askopenfilename(title='选择图片')  
		copyfile(path_,'./tmp/demo.jpg')
		path.set(path_)
		print('choosepic path_',path_)
		showpic(e1.get(),sLab)
		text.insert(END,'输入图片,%s: '%path_.split('/')[-1])
		showAnnoWidth(e1.get())

	def batchpic():
		clear()
		hidesingleorshow(True)
		path_ = askopenfilenames(title='批处理')
		print('batchpic',path_,type(path_))
		if checkVar.get()==0:
			text.insert(END,'基于Canny算法的检测\n')
		else:
			text.insert(END,'基于FCN的检测\n')
		for ph in path_:
			fn=ph.split('/')[-1]
			if checkVar.get()==0:
				w1=findCrack(ph)
			else:
				w1=findCrack_fcn(ph)
			text.insert(END,'%s\t%s\n'%(fn,w1))
			text.update()
			text.see(END)
		text.insert(END,'检测完成！\n')




	def showpic(path,slabel):  
	    img_open = Image.open(path)
	    w, h = img_open.size
	    img_open.thumbnail((717, 90),Image.ANTIALIAS) 
	    # resize()方法中的size参数直接规定了修改后的大小，而thumbnail()方法按比例缩小，size参数只规定修改后size的最大值。
	    # img_open = img_open.resize((512,64),Image.ANTIALIAS)
	    img=ImageTk.PhotoImage(img_open)  
	    slabel.config(image=img)  
	    slabel.image=img #keep a reference  
	    e1.pack_forget()
	
	def showAnnoWidth(o_name):
		a_name = o_name.replace('image','annotation')
		a_name = a_name.replace('jpg','png')
		w = getGapWidth(a_name)
		# lb['text'] = w
		text.insert(END,w)

	def showCrack1():
		# print(e1.get(),type(e1.get()),path.get())
		fname = e1.get()
		w1 = findCrack(fname,output='./tmp/dip.jpg')
		# lab_2_gap['text'] = w1
		text.insert(END,'\n基于Canny算法的检测：%s'%w1)

		showpic('./tmp/dip.jpg',sLab_dip_crack)
		

	def showCrack2():
		findCrack_fcn(e1.get())
		showpic('./tmp/fcn.png',sLab_fcn_crack)
		text.insert(END,'\n基于FCN的检测：')
		showAnnoWidth('./tmp/fcn.png')

	def solveCom(*args):
		print(comboxlist.get())

	root = Tk()
	root.title('手机壳合缝宽度异常检测')
	path = StringVar()
	radioValue=IntVar()
	comvalue = StringVar()
	checkVar = IntVar()
	SingleMode = True
	# cBtn = Button(root,text='选择图片',command=choosepic)
	# cBtn.grid(row=0,column=0)
	# fBtn = Button(root, text='检测缝隙1', command=showCrack1)
	# fBtn.grid(row=0, column=1)
	# fBtn2 = Button(root, text='检测缝隙2', command=showCrack2)
	# fBtn2.grid(row=0, column=2)	
	# lab_0 = Label(root,text='合缝最大宽度/像素')
	# lab_0.grid(row=0,column=6)
	e1 = Entry(root,state='readonly',text=path)
	# e1.grid(row=0, column=2)

	# lab_1 = Label(root,text='输入图片')
	lab_1 = Button(root,text='选择图片',command=choosepic)
	lab_1.grid(row=0,column=0)
	btn_p = Button(root,text='批处理',command=batchpic)
	btn_p.grid(row=0,column=4)
	sLab = Label(root) # 原始图片不动
	sLab.grid(row=1, column=0,columnspan=5)
	# lab_1_gap = Label(root)
	# lab_1_gap.grid(row=1,column=6)

	# lab_2 = Label(root,text='基于投影法的合缝检测')
	
	# rdioOne = Radiobutton(root,text='1',variable=radioValue, value=1)
	# rdioOne.grid(row=0,column=1)
	lab_2 = Button(root,text='DIP',command=showCrack1)
	lab_2.grid(row=0,column=1)
	sLab_dip_crack = Label(root) # 图像处理检测到的缝隙
	sLab_dip_crack.grid(row=2,column=0,columnspan=5)
	# lab_2_gap = Label(root)
	# lab_2_gap.grid(row=2,column=6)

	# lab_3 = Label(root,text='基于全卷积网络的合缝检测')
	# rdioTwo=Radiobutton(root,text='2',variable=radioValue,value=2)
	# rdioTwo.grid(row=0,column=3)

	lab_3 = Button(root,text='FCN',command=showCrack2)
	lab_3.grid(row=0,column=2)
	sLab_fcn_crack = Label(root)
	sLab_fcn_crack.grid(row=3,column=0,columnspan=5)
	# lab_3_gap = Label(root)
	# lab_3_gap.grid(row=3,column=6)
	# comboxlist = ttk.Combobox(root,textvariable=comvalue) 
	# comboxlist['values']=('DIP','FCN')
	# comboxlist.current(0)
	# comboxlist.bind("<<ComboboxSelected>>",solveCom)
	# comboxlist.grid(row=0,column=3,columnspan=1)

	C1 = Checkbutton(root,text='',variable=checkVar,onvalue=1,offvalue=0,height=1,width=2)
	C1.grid(row=0,column=3,columnspan=1)
	# 文本框
	text = tkst.ScrolledText(root,width=80,height=5)
	text.grid(row=4,column=0,columnspan=6)

	showpic('./tmp/0063.jpg',sLab) #预加载一个图片
	path.set('./tmp/0063.jpg')
	text.insert(END,'输入图片：')
	showAnnoWidth('./tmp/0063.jpg')
	# getData200Width()
	root.mainloop()