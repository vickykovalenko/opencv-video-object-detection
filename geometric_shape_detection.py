import cv2 
import numpy as np

filename = 'video.mp4'
output_filename = 'result_video.mp4'


def main():
	cap = cv2.VideoCapture(filename)

	fourcc = cv2.VideoWriter_fourcc(*'mp4v')

	output_frames_per_second = 20.0
	file_size = (1920,1080) # Assumes 1920x1080 mp4
	writer = cv2.VideoWriter(output_filename, fourcc, output_frames_per_second, file_size)

	key = None
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			frame = detect_shape(frame)
			writer.write(frame)
		else:
			break

	cap.release()
	writer.release()
	cv2.destroyAllWindows()

def detect_shape(frm):

	image = frm.copy()

	image_blur = cv2.GaussianBlur(image, (3, 3), 0)

	image_blur_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	#inRange() to generate the mask that has a value of 255 for pixels where the HSV values 
	#fall within the specified color range and a value of 0 for pixels whose values 
	#don’t lie in this interval

	lower_yellow = np.array([20, 43, 46])
	upper_yellow = np.array([80, 255, 255])
	yellow_mask = cv2.inRange(image_blur_hsv, lower_yellow, upper_yellow)
	
	image_blur_hsv = yellow_mask 

	img = np.zeros(frm.shape, np.uint8)
	# detect change in the image color and mark it as contour
	contours, _ = cv2.findContours(image_blur_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:

		# Find centroid 
		M = cv2.moments(contour)
		if M["m00"]!=0:
			cX = int(M["m10"]/M["m00"])
			cY = int(M["m01"]/M["m00"])
		else:
			cX, cY = 0, 0

		# Compute perimeter of contour and perform contour approximation
		peri = cv2.arcLength(contour, True)
		fig = cv2.approxPolyDP(contour, 0.03*peri, True) # approximation of polygons


		# get rectangle bounding contour
		[x, y, w, h] = cv2.boundingRect(contour)

		# discard areas that are too small
		if w<40 or h<40:
			continue

		if len(fig) == 3:
			text_color_triangle = (36,255,12)
			cv2.putText(img, 'triangle', (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color_triangle, 2)
		elif len(fig) == 4:
			text_color_rectangle = (25, 255, 255)
			cv2.putText(img, 'rectangle', (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color_rectangle, 2)
		else:
			text_color_circle = (160,100,20)
			cv2.putText(img, 'circle', (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color_circle, 2)

	image_blur_hsv = cv2.addWeighted(frm, 0.5, img, 0.5, 0.0) # blend images
	return image_blur_hsv

main()