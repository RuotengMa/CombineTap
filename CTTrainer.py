
# coding: utf-8


import keyboard
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import pytesseract as pyt
pyt.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'





def checkContours(hier,cont,hiercounter,contours,hierarchy,usedList,edges,frame):

    nos_flag = 0
    sib_flag = 0
    level = 4
    nos = 0
    ocont = cont
    tempList = [hiercounter]
    if hiercounter in usedList:
        return [-1], -1, usedList
    while(level >= 2):
        ccont = cont
        chier = hier
        if hier[2] >= 0: #has child
            nhier = hierarchy[0][hier[2]]
            ncont = contours[hier[2]]
            nindex = hier[2]
            if hier[2] in usedList:
                return [-1], -1, usedList
            tempList.append(chier[2])
            if sameContour(cont,ncont,75):
                tempList.append(nhier[2])
                nhier = hierarchy[0][nhier[2]]
                ncont = contours[nhier[2]]
            else:
                ccont = cont
        elif level != 0:
            return [-1], -1, usedList
        if level == 3:
            if not checkSquare(ccont):
                return [-1], -1, usedList
        if level == 2:
            if checkSibling(chier,hierarchy) > 1:
                sib_flag = 1
        
            if sib_flag == 0:
                nos,tempList = checkNOS(chier,ccont,tempList,hierarchy,contours,usedList,edges,frame)
                if nos == -1:
                    return [-1], -1, usedList
            elif sib_flag == 1:
                nos,tempList = getSibNos(chier,ccont,tempList,hierarchy,contours,usedList,edges,frame)
                if nos == -1:
                    return [-1], -1, usedList
                break

        level = level - 1
        hier = nhier
        cont = ncont

    usedList = usedList + tempList
    return ocont, nos, usedList
def getSibNos(hier,ccont,tempList,hierarchy,contours,usedList,edges,frame):
    tcombineID,tempList = checkNOS(hier,ccont,tempList,hierarchy,contours,usedList,edges,frame)
    if tcombineID == -1:
        return -1, tempList
    combineID = str(tcombineID)
    lhier = hier[0]
    rhier = hier[1]
    while lhier >= 0:
        tcombineID,tempList = checkNOS(hierarchy[0][lhier],contours[lhier],tempList,hierarchy,contours,usedList,edges,frame)
        if tcombineID == -1:
            return -1, tempList
        combineID = combineID + ',' + str(tcombineID)
        lhier = hierarchy[0][lhier][0]
    while rhier >= 0:
        tcombineID,tempList = checkNOS(hierarchy[0][rhier],contours[rhier],tempList,hierarchy,contours,usedList,edges,frame)
        if tcombineID == -1:
            return -1, tempList
        combineID = combineID + ',' + str(tcombineID)
        rhier = hierarchy[0][rhier][1]
    return combineID, tempList
def checkNOS(hier,ccont,tempList,hierarchy,contours,usedList,edges,frame):
    currentcont = ccont
    tempflag = checkShape(currentcont)
    level = 2
    nos_flag = 0
    nos = -1
    while(level >= 0):
        ccont1 = currentcont
        chier1 = hier
        if hier[2] >= 0: #has child
            nhier1 = hierarchy[0][hier[2]]
            ncont1 = contours[hier[2]]
            nindex1 = hier[2]
            if hier[2] in usedList:
                return -1,tempList
            tempList.append(chier1[2])
            if sameContour(currentcont,ncont1,75):
                tempList.append(nhier1[2])
                nhier1 = hierarchy[0][nhier1[2]]
                ncont1 = contours[nhier1[2]]
            else:
                ccont1 = currentcont
        elif level != 0:
            return -1,tempList
        if level == 2:
            if checkShape(ccont1) == 0:
                nos_flag = 1
            if checkShape(ccont1) == 2:
                nos_flag = 2
                
        if level == 1:
            if checkSibling(chier1,hierarchy) > 1:
                return -1,tempList
        if level == 0:
            if nos_flag == 2:
                nos = 'n'+str(checkSibling(chier1,hierarchy))
                if nhier1[2] > 0:
                    return -1,tempList
            if nos_flag == 1:
                ocroi = getROI(ccont1,edges,frame)
                if ocroi.size == 1:
                    return -1,tempList
                ocroi = cv2.flip(ocroi, 0)
                ocroi = cv2.cvtColor(ocroi, cv2.COLOR_BGR2GRAY)
                cv2.imshow('sd',ocroi)
                letter = checkOCR(ocroi)
                nos = letter
        level = level - 1
        hier = nhier1
        currentcont = ncont1
    return nos, tempList
def checkOCR(img):
    return pyt.image_to_string(img, lang = 'eng', config='--psm 10')
def getROI(cont,edges,frame):
    contour_mask = np.zeros_like(edges)
    cv2.drawContours(contour_mask, [cont], -1, 255, -1)
    (x, y) = np.where(contour_mask == 255)
    if x.size == 0 or y.size ==0:
        return [0]
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    ocroi = frame[topx:bottomx+1, topy:bottomy+1]
    return ocroi
def sameContour(cont,ncont,thre):
    dcont = cv2.contourArea(cont) - cv2.contourArea(ncont)
    if abs(dcont) <= thre:
        return True
    else:
        return False
def checkSibling(hier,hierarchy):
    numOfSib = 1
    lhier = hier[0]
    rhier = hier[1]
    while lhier >= 0:
        numOfSib = numOfSib + 1
        lhier = hierarchy[0][lhier][0]
    while rhier >= 0:
        numOfSib = numOfSib + 1
        rhier = hierarchy[0][rhier][1]
    return numOfSib
def checkSquare(cnt):
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx) == 4:
        return True
    else:
        return False
def checkShape(cnt):
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx) >= 4 and len(approx) < 8:
        return 0
    if len(approx) == 3:
        return 1
    if len(approx) > 8:
        return 2
def getMarker():
    camera = cv2.VideoCapture(0)

    # reduce frame size to speed it up
    w = 1280
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
    start = False
    markerID = -1
    # capture loop
    while True:

        # get frame
        ret, frame = camera.read()

        # get frame size
        w, h = frame.shape[:2]

        # mirror the frame 
        frame = cv2.flip(frame, 1)


        if cv2.waitKey(5) == 32:
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = cv2.GaussianBlur(mask,(5,5),0)
            edges = cv2.Canny(mask,100,200)
            im1, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            toDraw = []
            markerID = []
            usedList = []
            hiercounter = 0
            for conthier in zip(contours, hierarchy[0]):
                cont = conthier[0]
                hier = conthier[1]
                if(checkShape(cont) == 0): #no child
                #get three level up
                #check for requirements
                    cont, nos, usedList = checkContours(hier,cont,hiercounter,contours,hierarchy,usedList,edges,frame)
                    if len(cont) > 1:
                        toDraw.append(cont)
                        markerID = nos
                hiercounter += 1
            cv2.drawContours(frame, toDraw, -1, (0,255,0), 2)
            for i in range (len(toDraw)):
                    M = cv2.moments(toDraw[i])
                    if M['m00'] == 0:
                        continue
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.putText(frame,str(markerID),(cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)


            cv2.imshow('frame1', frame)
        cv2.imshow('frame', frame)
        # exit on ESC press
        if cv2.waitKey(5) == 27:
            break

        # clean up
    cv2.destroyAllWindows()
    camera.release()
    cv2.waitKey(1) # extra waitKey sometimes needed to close camera window
    return str(markerID)



wfile = open('keyToID.txt','a')
Aflag = 0
while True:
    while True:
        flag = 0
        print('Please press the keyboard button you want to map')
        a = keyboard.read_hotkey()
        a = keyboard.read_hotkey()
        print('The key you pressed is ' + str(a))
        print('To re-select a button, press Enter; to confirm the selection, press Esc')
        while True:
            if keyboard.is_pressed('enter'):
                break
            elif keyboard.is_pressed('esc'):
                print('Keyboard input finished')
                flag = 1
                break
        if flag == 1:
            break
    while True:
        breakflag = 0
        print('Please put your custom button under the camera')
        input("Press Enter to continue...")
        obtainedID = getMarker()
        print('The marker ID obtained is: ' + str(obtainedID))
        print('Press enter to retry, Press ESC to continue')
        while True:
            if keyboard.is_pressed('enter'):
                break
            elif keyboard.is_pressed('esc'):
                print('Keyboard input finished')
                breakflag = 1
                break
        if breakflag == 1:
            break
    print('For key ' + str(a) + ', the marker is ' + str(obtainedID))
    wfile.write(str(obtainedID) + ',' + str(a) + '\n')
    control = input('To continue input, press Y; to exit, press N:')
    while True:
        if control == 'y':
            break
        elif control == 'n':
            print('Keyboard input finished')
            Aflag = 1
            break
    if Aflag == 1:
        break
wfile.close()           


