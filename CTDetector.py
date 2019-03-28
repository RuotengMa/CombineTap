
# coding: utf-8



import keyboard
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import re
import subprocess
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
    ocroi = 0
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
            if nos_flag == 1:
                ocroi = getROI(ccont1,edges,frame)
                if ocroi.size == 1:
                    return -1,tempList
                ocroi = cv2.flip(ocroi, 0)
                ocroi = cv2.cvtColor(ocroi, cv2.COLOR_BGR2GRAY)
                if nos_flag == 1:
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
def checkandperform(selectedID, idDict):
    sid = str(selectedID).split(',')
    if sid[0] == 'n9':
        subprocess.run(["python","control.py"])
        return 1
    first = []
    second = []
    for a in sid:
        if bool(re.search(r'n\d',a)):
            first.append(a)
        else:
            second.append(a)
    sid = first + second

    length = len(sid)
    fw = idDict.get(str(sid[0]),0)
    if fw == 0:
        return 0
    press = fw.split('\n')[0]
    if length > 1:
        for i in sid[1:length]:
            ws = idDict.get(str(i),0)
            if ws == 0:
                return 0
            press = press + '+' + ws.split('\n')[0]
    print(press)
    keyboard.send(press)
    return 1
def getMarker(idDict):
    camera = cv2.VideoCapture(0)

    # reduce frame size to speed it up
    w = 1280
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, w * 3/4) 
    allIDs = []
    pmarkerID = []
    min_HSV = np.array([0, 30, 51], dtype = "uint8")
    max_HSV = np.array([14, 213, 223], dtype = "uint8")
    kernel = np.ones((3,3))
    selected = -1
    selecount = 0
    checking = True
    checktimer = 0
    toDraw = []
    while True:
        # get frame
        ret, frame = camera.read()

        # get frame size
        w, h = frame.shape[:2]

        # mirror the frame (my camera mirrors by default)
        frame = cv2.flip(frame, 1)
        output = frame
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        skinRegionYCrCb = cv2.inRange(imageYCrCb,min_HSV,max_HSV)

        skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)
        skinYCrCb = cv2.cvtColor(skinYCrCb, cv2.COLOR_BGR2GRAY)
        im2, contours2, hierarchy2 = cv2.findContours(skinYCrCb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours2) > 0:
            c = max(contours2, key = cv2.contourArea)
            epsilon = 0.1*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            hull = cv2.convexHull(approx)
            ip = -1
            yp = -1
            for i in approx:
                y = i[0][1]
                if y > yp:
                    yp = y
                    ip = i
        
        if checking:
            checktimer = checktimer + 1
            if checktimer == 10:
                pmarkerID = []
                mask = cv2.GaussianBlur(mask,(5,5),0)
                edges = cv2.Canny(mask,100,200)
                im1, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                valid = []
                toDraw = []
                markerID = []
                usedList = []
                hiercounter = 0
                for conthier in zip(contours, hierarchy[0]):
                    cont = conthier[0]
                    hier = conthier[1]
                    if(checkShape(cont) == 0): #no child
                        cont, nos, usedList = checkContours(hier,cont,hiercounter,contours, hierarchy,usedList,edges,frame)
                        if len(cont) > 1:
                            valid.append(cont)
                            markerID.append(nos)
                            allIDs.append(nos)
                    hiercounter += 1
                for i in range (len(valid)):
                        M = cv2.moments(valid[i])
                        if M['m00'] == 0:
                            continue
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])

                        cv2.putText(output,str(markerID[i]),(cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                cv2.drawContours(output, valid, -1, (0,255,0), 2)
                toDraw = valid
                pmarkerID = markerID
                checktimer = 0
            else:
                for i in range (len(toDraw)):
                        M = cv2.moments(toDraw[i])
                        if M['m00'] == 0:
                            continue
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])

                        cv2.putText(output,str(pmarkerID[i]),(cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                cv2.drawContours(output, toDraw, -1, (0,255,0), 2)
            if cv2.waitKey(5) == 32:
                checking = False
                checktimer = 0
                cv2.imshow('savedFrame', output)
                savedToDraw = toDraw
                savedMarker = pmarkerID
        else:
            cv2.circle(output, tuple(ip[0]), 6, (0,255,255), 2)
            for i in range (len(savedToDraw)):
                M = cv2.moments(savedToDraw[i])
                if M['m00'] == 0:
                    continue
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.putText(output,str(savedMarker[i]),(cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                ioe = cv2.pointPolygonTest(savedToDraw[i], tuple(ip[0]), False)
                
                if ioe > 0:
                    if selected == i:
                        selecount = selecount + 1
                    selected = i
                    if selecount > 15:
                        selectedID = savedMarker[selected]
                        suc = checkandperform(selectedID, idDict)
                        if suc == 0:
                            print('Unknown Button')
                        selecount = 0
            cv2.drawContours(output, savedToDraw, -1, (0,255,0), 2)


            if cv2.waitKey(5) == 32:
                    checking = True
                    savedToDraw = []
                    savedMarker = []        
                    checktimer = 0
                    

        cv2.imshow('frame', output)   
        # exit on ESC press
        if cv2.waitKey(5) == 27:
            break

    # clean up
    cv2.destroyAllWindows()
    camera.release()
    cv2.waitKey(1)


rfile = open('keyToID.txt','r')
idDict = dict()
for line in rfile:
    kvPairs = line.split(',')
    idDict[kvPairs[0]] = kvPairs[1]

getMarker(idDict)
