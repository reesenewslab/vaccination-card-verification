# import the necessary packages
import numpy as np
import imutils
import cv2

def align_images(image, template, fileTag, maxFeatures=500, keepPercent=0.2,
	debug=False):
	'''
	Aligns image to template via the steps:
		1. ORB keypoint detection
		2. Brute force hamming distance keypoint matching
		3. RANSAC homography estimation

	Input:
		image - the image to be aligned
		template - the template the image should be aligned to
		fileTag - a prefix to be used on the debug output files
		[Optional] maxFeatures - the maximum number of keypoints to detect
		[Optional] keepPercent - the percentage of keypoint matches to actually use during the homography estimation

	Output: (aligned, RANSAC_inliers)
		aligned - the aliged image
		RANSAC_inliers - the number of keypoint matches which are determined as inliers for the RANSAC homography estimation
	'''

	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

	# use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)

	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)

	# sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)

	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]

	# allocate memory for the keypoints (x,y-coordinates) from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")

	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt

	# compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=5)
	mask_inliers = np.array(mask)
	mask_outliers = np.logical_not(mask_inliers).astype(int)

	print(f'matches: {len(matches)}, output mask: {mask.shape}, mask_inliers.count_nonzero(): {np.count_nonzero(mask_inliers)}, mask_outliers.count_nonzero(): {np.count_nonzero(mask_outliers)}')

	# check to see if we should visualize the matched keypoints
	if debug:
		# draw matches (RANSAC inliers)
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None, matchColor=(0, 255, 0), matchesMask=mask_inliers)

		# draw matches (RANSAC outliers)
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, matchedVis, matchColor=(0, 0, 255), matchesMask=mask_outliers, flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)

		# draw unmatched
		# matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
		# 	matches, matchedVis, matchColor=(0, 0, 255), matchesMask=mask_outliers, flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)

		matchedVis = imutils.resize(matchedVis, width=1000)
		# cv2.imshow("Matched Keypoints", matchedVis)
		# cv2.waitKey(0)
		cv2.imwrite(f"output/{fileTag}_matched_keypoints.jpg", matchedVis)

	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))

	# return the aligned image
	return aligned, np.count_nonzero(mask_inliers)
