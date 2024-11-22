print('[STATUS] Import packages')

from tensorflow.keras.models import load_model

# import necessarry python Files
from Algorithm_Mask_Prediction import *
from Useful_Tools import *
from Computer_Vision_Tools import *
from Federated_Data_Colection import *
from datetime import datetime

#necessarry paths

pathToProtoModel = pathToProcess(r"/Computer Vision/caffeFaceModel/deploy.prototxt")
pathToCaffeWeightsModel = pathToProcess(r"/Computer Vision/caffeFaceModel/res10_300x300_ssd_iter_140000.caffemodel")
pathToMaskModel =  pathToProcess(r"/Computer Vision/FLModel.h5")
pathFederated =  pathToProcess(r'/dataCollection')
pathNoCrop = pathToProcess(r'/dataCollection/NoCrop')
pathNoMask = pathToProcess(r'/dataCollection/NoMask') 
pathWithMask = pathToProcess(r'/dataCollection/WithMask')
pathDeleteNoCrop = pathToProcess(r'/dataCollection/NoCrop')

#create dataColection folders

folderStructureCreation(pathFederated)
folderStructureCreation(pathNoCrop)
folderStructureCreation(pathNoMask)
folderStructureCreation(pathWithMask)

# load our serialized face detector model from disk
faceNet = cv2.dnn.readNet(pathToProtoModel, pathToCaffeWeightsModel)


print (pathToMaskModel)

maskNet = load_model(pathToMaskModel) #load_model(args["model"])
print ('[STATUS] Load models')

# initialize the video stream and allow the camera sensor to warm up
VideoStream = getVideoStream()



listNameMask = []
listNameNoMask = []
dateFileName =  datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# loop over the frames from the video stream

while True:
    
    
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    
    getFrames = getFramesFromCamera(VideoStream)
    (locs, preds) = detect_and_predict_mask(getFrames, faceNet, maskNet)
    
    # loop over the detected face locations and their corresponding
    # locations
    
    for (box, pred) in zip(locs, preds):
        
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = getBorderCoordinates(box)

        (mask, withoutMask) = getTheProbabilityOfWearingTheMask(pred)
        
        # determine the class label and color we'll use to draw
        # the bounding box and text
        
        (label, color) = getLabelForMask(mask, withoutMask)
        
        tupleCoordinates = (startX, startY, endX, endY)
        (photoName, MaskName, NoMaskName) = SaveFederatedDataColection(label, getFrames, tupleCoordinates)
        if MaskName == True:
            date = datetime.now()
            listNameMask.append("Mask " + "PhotoName: "+ str(photoName) + " DateAndTime: " + str(date) + "\n")            
        if NoMaskName == True:
            date = datetime.now()
            listNameNoMask.append("NoMask " + "PhotoName: "+ str(photoName) + " DateAndTime: " + str(date) +"\n")
        
        setColorAndTextToBorder(mask, withoutMask, getFrames, startX, startY, endX, endY)       
             
    key = UserInterfaceSettings(getFrames)    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
ShutDownTheSystem(VideoStream)

#Delete NoCrop directory
deleteUnnecessaryFolder(pathDeleteNoCrop)


outputData = pathToProcess(r'/dataCollection/outputData_' + str(dateFileName) + '.txt')
dateEnd = datetime.now()

with open(outputData, "w+") as out_file:
    for item in listNameMask:
        out_file.write(item)
    out_file.write("\n")
    out_file.write('[SUMMARY] : From date: ' + str(dateFileName) + "\n")
    out_file.write('[SUMMARY] : To date:   ' + str(dateEnd) + "\n")
    out_file.write('[SUMMARY] : People(s) detected with mask: ' + str(len(listNameMask)) + "\n") 
    out_file.write("\n")
    
    for item in listNameNoMask:
        out_file.write(item)
    out_file.write("\n")
    out_file.write('[SUMMARY] : From date: ' + str(dateFileName) + "\n")
    out_file.write('[SUMMARY] : To date:   ' + str(dateEnd) + "\n")
    out_file.write('[SUMMARY] : People(s) detected without mask: ' + str(len(listNameNoMask)) + "\n") 
    out_file.write("\n")