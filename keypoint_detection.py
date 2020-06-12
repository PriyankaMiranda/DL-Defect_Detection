import cv2 
import matplotlib.pyplot as plt
import numpy as np

#reading image
img1 = cv2.imread('assets/good_image.png')  

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#reading image
img2 = cv2.imread('assets/bad_val_data/image_0_9404.jpg')  
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.xfeatures2d.SIFT_create(nfeatures = 100000000)
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)
# computing a homography requires at least 4 matches
if len(matches) > 4:
	# construct the two sets of points
	ptsA = np.float32([keypoints_1[i].pt for i in range(len(matches))])
	ptsB = np.float32([keypoints_2[i].pt for i in range(len(matches))])
	# compute the homography between the two sets of points
	(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,4.0)
		
result = cv2.warpPerspective(img1, H,(img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

(hA, wA) = img1.shape[:2]
(hB, wB) = img2.shape[:2]
vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
vis[0:hA, 0:wA] = img1
vis[0:hB, wA:] = img2
# loop over the matches

n=0
for x in matches:
	if (status[n]):
		ptA = (int(keypoints_1[x.queryIdx].pt[0]), int(keypoints_1[x.queryIdx].pt[1]))
		ptB = (int(keypoints_2[x.trainIdx].pt[0]) + wA, int(keypoints_2[x.trainIdx].pt[1]))
		cv2.line(vis, ptA, ptB, (255, 255, 255), 1)
	n=n+1

# for ((trainIdx, queryIdx), s) in zip(matches, status):
# 	# only process the match if the keypoint was successfully
# 	# matched
# 	if s == 1:
# 		# draw the match
# 		ptA = (int(keypoints_1[queryIdx][0]), int(keypoints_1[queryIdx][1]))
# 		ptB = (int(keypoints_2[trainIdx][0]) + wA, int(keypoints_2[trainIdx][1]))
# 		cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
plt.imshow(vis)
plt.show()
plt.imshow(result)
plt.show()


# vals=[]
# keypoints_not_matched = []
# descriptors_not_matched = []

# for m in matches:
# 	vals.append(m.queryIdx)

# max_len = len(descriptors_1)
# if (len(descriptors_2)>len(descriptors_1)):
# 	max_len = len(descriptors_2)

# for x in range(max_len):
# 	if x not in vals:
# 		try:
# 			keypoints_not_matched.append(keypoints_1[x])
# 			descriptors_not_matched.append(descriptors_1[x])
# 		except:
# 			keypoints_not_matched.append(keypoints_2[x])
# 			descriptors_not_matched.append(descriptors_2[x])
# DMatch.distance - Distance between descriptors. The lower, the better it is.
# DMatch.trainIdx - Index of the descriptor in train descriptors
# DMatch.queryIdx - Index of the descriptor in query descriptors
# DMatch.imgIdx - Index of the train image.




# img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
# plt.imshow(img3)
# plt.show()


# img_2 = cv2.drawKeypoints(gray2,keypoints_not_matched,img2)
# plt.imshow(img_2)
# plt.show()

