import cv2
import numpy as np
import os
import pickle

from keras.models import load_model
from keras.preprocessing import image
#Usingg the following statement to avoid getting warnings in the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Setting up the initial height and width as we set during the odel designing
img_width, img_height = 640, 480
#Loading the saved model
model = load_model('Models/MODEL1.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def nothing(x):
    #This function is used for enabling Trackbars.
    pass

def Predict_Number():
    #Here we get the saved image, in which we just drew on air...
    img = image.load_img("test.jpg", target_size=(img_width, img_height))
    #Making appropriate changes to image so it can be passed to predict method...
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    #Here we can use predict() method as predict_classes will no longer be available in future...
    #We can use np.argmax() function to get class labes as by softmax activation we get probablities of each class...
    classes = model.predict_classes(images, batch_size=10)
    #We are predicting numbers, And each number from 0-9 are assigned to same class labels...
    #Hence for 0 we get output as [0], So no need to change the class labesls using the pickle...
    print(classes)

#Capturing the video using webcam...
cap = cv2.VideoCapture(0)

'''
This is to create trackbar for changing the upper and lower limits of HSV values.
These range of HSV values can be used to seperate color from other colors in background.
cv2.namedWindow("Trackbar")
cv2.createTrackbar("L - H","Trackbar",0,179,nothing)
cv2.createTrackbar("L - S","Trackbar",0,255,nothing)
cv2.createTrackbar("L - V","Trackbar",0,255,nothing)
cv2.createTrackbar("U - H","Trackbar",0,179,nothing)
cv2.createTrackbar("U - S","Trackbar",0,255,nothing)
cv2.createTrackbar("U - V","Trackbar",0,255,nothing)
'''

#Tracking variable tells if we are writing or not. Using space we can toggle this value.
#Once the tracking is completed, image is saved...
#For this Tracked variable is used...
Tracking = False
Tracked = False

#Old point stores the coordinates of previous tracked points...
#These points help in plotting the line we drew...
#This in turn making it appear that we wrote something and it stays accross frames...
old_pt = ()
new_pt = ()

oldPoints = []
newPoints = []

x = 0

while True:
    #These are the identified, limits for a specific shade of blue to identify the tip of my pen...
    #By experimenting with these values, we can seperate colors from some range from others...
    lower_blue = np.array([95,106,45])
    upper_blue = np.array([179,255,255])

    '''
    l_h = cv2.getTrackbarPos("L - H","Trackbar")
    l_s = cv2.getTrackbarPos("L - S","Trackbar")
    l_v = cv2.getTrackbarPos("L - V","Trackbar")
    u_h = cv2.getTrackbarPos("U - H","Trackbar")
    u_s = cv2.getTrackbarPos("U - S","Trackbar")
    u_v = cv2.getTrackbarPos("U - V","Trackbar")

    lower_blue = np.array([l_h,l_s,l_v])
    upper_blue = np.array([u_h,u_s,u_v]
    
    '''

    ret, frame1   = cap.read()
    #We flip the captured image so users can see a mirror image of themself...
    #If not flipped, We will see offosite movement in camera...
    frame1 = cv2.flip(frame1, 1)
    #Displaying the captured frame...
    cv2.imshow("frame",frame1)

    #Converting to HSV color mode...
    hsv1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)

    #Masking the image (Limiting the colors of image to specific range...)
    mask1 = cv2.inRange(hsv1,lower_blue,upper_blue)

    #Reading the next frame...
    ret, frame2   = cap.read()
    frame2 = cv2.flip(frame2, 1)

    hsv2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)

    mask2 = cv2.inRange(hsv2,lower_blue,upper_blue)
    
    #Now calculating the absolute diff, of both the masks, hence reducing most of the unwanted areas
    mask = cv2.absdiff(mask1,mask2)

    #Setting up kernal size, more size = more pixelated mask image
    kernel = np.ones((5,5), np.uint8)
    #Dilating the image means we increase the unmasked areas giving better results...
    result = cv2.dilate(mask, kernel, iterations=1) 

    #Finding the edges of unmasked part
    edge = cv2.Canny(result,30,200)
    #Getting all the contours present in the image
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    #Setting up frame  as frame 1 to display only 1 frame...
    frame = frame1

    for c in contours:
        #In the biggest contour, we find the centroid...
        (x,y,w,h) = cv2.boundingRect(c)
        Centroid = (int((x + x + w) /2), int((y + y + h) /2))
        YCentroid= (y+y+h)/2
        #That centrois is then used to plot a filled circle, as it appears that out pen tip is identified...
        frame = cv2.circle(frame, Centroid, 1, (0, 255, 0), 8)
        
        #Keeping the track of previous and current point...
        #@ points are needed, because we want to plot a line between them
        old_pt = new_pt
        new_pt = Centroid
        if Tracking and old_pt != () and new_pt != ():
            #Once we start talking, We fill the arrays and plot that array on each new frame...
            oldPoints.append(old_pt)
            newPoints.append(new_pt)
            for i in range(len(oldPoints)):
                #looping over each point to plot line between them...
                frame = cv2.line(frame, oldPoints[i], newPoints[i], (0, 0, 255), 4)
                result = cv2.line(result, oldPoints[i], newPoints[i], (255, 255, 255), 10)
        #print(Centroid)

    #mask = cv2.blur(mask,(10,10))

    #result = cv2.bitwise_and(frame1,frame1,mask=mask)
    if Tracking:
        #Just to indicate that we are tracking, a border of yellow appears when Tracking = True
        cv2.rectangle(frame, (0,0), (640,480), (0,255,255), 3)

    #cv2.imshow("frame",frame)
    #Printing out the result, i.e. what we draw
    cv2.imshow("Result",result)
    

    if cv2.waitKey(1) & 0xFF == ord(" "):
        #This block runs when we hit space...
        if Tracking == False:
            Tracking = True
            #print("Tracking")
        else:
            Tracking = False
            #print("!Tracking")
            oldPoints = []
            newPoints = []
            Number_Image = result
            cv2.imwrite("test.jpg",Number_Image)
            Predict_Number()
            x+=1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()