from imutils.video import VideoStream
import imutils
import time
import cv2

# initialize the video stream and allow the camera sensor to warm up

def getVideoStream():
    #vs = VideoStream(src=0, framerate = 5).start()
    vs = VideoStream(src=0).start()
    #time.sleep(2.0)
    return vs

# grab the frame from the threaded video stream and resize it
# to have a maximum width of 400 pixels
    
def getFramesFromCamera(VideoStream):
    #vs = getVideoStream()
    frame = VideoStream.read()
    frame = imutils.resize(frame, width=400)
    return frame

# initialize the video stream and allow the camera sensor to warm up
print("[STATUS] Start video stream")

# unpack the bounding box 
def getBorderCoordinates(coordinates):
        (startX, startY, endX, endY) = coordinates
        #print (box)
        return startX, startY, endX, endY
    
# unpack the mask predictions
def getTheProbabilityOfWearingTheMask(prediction):
        (mask, withoutMask) = prediction
        return mask, withoutMask
    

def getLabelForMask(mask, withoutMask):
    print ("Mask   ",mask)
    print ("NoMask ",withoutMask)
    print("\n")
    label = "Undefined"
    color = (0,0,0)
    label = "MaskNotOk"
    color = (255,140,0)
    if (withoutMask < 0.500 ) and (mask < 0.900):
        label = "MaskNotOk"
        color = (255,140,0)
    elif mask > 0.870 and withoutMask < 0.5:
        label = "Mask"
        color = (0,255,0)
    elif mask< 0.5 and withoutMask > 0.65:
        label = "NoMask"
        color =  (0,0,255)
    if label is None:
        label = "MaskNotOk"
    if color is None:
        color = (255,140,0)

    return (label, color)
    

def setColorAndTextToBorder(mask, withoutMask, getFrames, startX, startY, endX, endY):
    
    (label, color) = getLabelForMask(mask, withoutMask)
    
    label = "{}: {:.2f}%".format(label, max(mask,  withoutMask) * 100)

    cv2.putText(getFrames, label, (startX, startY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(getFrames, (startX, startY), (endX, endY), color, 2)

def UserInterfaceSettings(getFrames):
    cv2.imshow("COVID-19 Mask Detector", getFrames)
    key = cv2.waitKey(1)
    return key
    
def ShutDownTheSystem(VideoStream):
    cv2.destroyAllWindows()
    VideoStream.stop()
