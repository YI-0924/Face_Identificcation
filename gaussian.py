# blur
# Importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# the larger the k is, the more blurred the output face will be
k = input("Gaussian parameter k (must be odd) : ")
k = int(k)

# A function for plotting the images
def plotImages(img):
	plt.imshow(img, cmap="gray")
	plt.axis('off')
	plt.style.use('seaborn')
	plt.show()

list1 = load_images_from_folder('img/img_layer2')
num=0
#item means img
for image in list1:
	#print(item)
        
	# Reading an item using OpenCV
	# OpenCV reads images by default in BGR format
	#image = cv2.imread('my_img1.jpg')

	# Converting BGR image into a RGB image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# plotting the original image
	#plotImages(image)
	#image = cv2.GaussianBlur(image, (k, k), 30)

	face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	face_data = face_detect.detectMultiScale(image, 1.1, 5)

	# Draw rectangle around the faces which is our region of interest (ROI)
	for (x, y, w, h) in face_data:
		#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		roi = image[y:y+h, x:x+w]

		# applying a gaussian blur over this new rectangle area
		roi = cv2.GaussianBlur(roi, (k, k), 30)

		# impose this blurred image on original image to get final image
		image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

	# Display the output
	cv2.imwrite('img'+str(num)+'.jpg', image)
	num+=1
	plotImages(image)