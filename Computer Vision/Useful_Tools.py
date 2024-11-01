import pathlib
from pathlib import Path
import os
import shutil

def pathToProcess(addToPath):
    #pathToProcess =  str(pathlib.Path(__file__).parent.absolute())
    pathToProcess = os.path.dirname(os.path.abspath(__file__))
    processedPath = pathToProcess.replace(r'/Computer Vision', '')
    processedPath = processedPath + addToPath
    return processedPath


def folderStructureCreation(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Creation of folder %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)
        

def deleteUnnecessaryFolder(path):
    try:
        shutil.rmtree(path)
    except OSError:
        print ("Deletion of the directory %s failed" % path)
    else:
        print ("Successfully deleted the directory %s" % path)
