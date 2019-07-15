from cv2 import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("news.jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# create an object where the face coordinates are detected
# where: 
#   1.1 is a decrease of 10% each time a face is searched. Tweak this
#   number to fine tune your results
faces = face_cascade.detectMultiScale(gray_img, 
scaleFactor=1.1,
minNeighbors=5)

print(type(faces))
print(faces)

# what faces will return is an array like the following:
# [[157  84 379 379]]
# 157 represents the x coordinate
# 84 represents the y coordinate
# 379 represents the width
# 379 represents the height
# for those coordinates, now draw a rectangle around the face
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),3)

resized = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

cv2.imshow("Gray",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()