import cv2
import numpy as np

detector=cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
trainImg=cv2.imread("stop_sign.jpg")
#trainImg = cv2.resize(trainImg, (0,0), fx=0.5, fy=0.5)
trainImgGray = cv2.cvtColor(trainImg,cv2.COLOR_BGR2GRAY)
trainKP,trainDesc=detector.detectAndCompute(trainImgGray,None)
cam=cv2.VideoCapture(0)
frame = 0
while True:
	ret, QueryImgBGR=cam.read()
	displayImgBGR = cv2.resize(QueryImgBGR, (0,0), fx=0.6, fy=0.6)
	QueryImgBGR = cv2.resize(displayImgBGR, (0,0), fx=0.3, fy=0.3)

	frame = frame + 1
	if(True):
		frame =0
		QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
		queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
		bf = cv2.BFMatcher()
		matches=bf.knnMatch(queryDesc,trainDesc, k=2)
		goodMatch=[]
		for m,n in matches:
			if(m.distance<0.75*n.distance):
				goodMatch.append(m)
		MIN_MATCH_COUNT=30
		if(len(goodMatch)>=MIN_MATCH_COUNT):
			tp=[]
			qp=[]
			for m in goodMatch:
				tp.append(trainKP[m.trainIdx].pt)
				qp.append(queryKP[m.queryIdx].pt)
			tp,qp=np.float32((tp,qp))

			H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
			h,w,c=trainImg.shape
			trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
			queryBorder=cv2.perspectiveTransform(trainBorder,H)
			cv2.polylines(displayImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
		#else:
		#	print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
		cv2.imshow('result',QueryImgBGR)
		if cv2.waitKey(10)==ord('q'):
			break
	else:
		cv2.imshow('result',QueryImgBGR)


cam.release()
cv2.destroyAllWindows()