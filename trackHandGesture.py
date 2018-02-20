import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# code assumes that the largest contour in the webcam frame is the hand, so need a clear background
# also requires proper lighting

def ReadData():
    # Data in format [B G R Label] from
    data = np.genfromtxt('./Skin_NonSkin.txt', dtype=np.int32)

    # the first value in data array (:) gets all rows, the second value (3) gets the fourth column
    labels = data[:,3]
    # get all rows and the first 3 columns (at indexes 0,1,2)
    data = data[:,0:3]

    return data, labels


def TrainTree(data, labels):
    # data shape is one long array with 3 channels (245057,3)
    # need to shape into 3 number array for cvtColor function
    # bgr has shape (245057,1,3)  (rows, col, chan)
    bgr = np.reshape(data,(data.shape[0],1,3))
    hsv = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    # once we have converted to HSV, we reshape back to original shape of (245057,3)
    dataHSV = np.reshape(hsv,(hsv.shape[0],3))

    trainData, testData, trainLabels, testLabels = train_test_split(dataHSV, labels, test_size=0.20, random_state=42)

    # Initialize our classifier
    gnb = GaussianNB()
    clf = gnb.fit(trainData, trainLabels)

    return clf


def detectSkin(frame):
    # reshape the frame to follow format of training data (rows*col, 3)
    data = np.reshape(frame, (frame.shape[0] * frame.shape[1], 3))
    bgr = np.reshape(data, (data.shape[0], 1, 3))
    hsv = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    # once we have converted to HSV, we reshape back to original shape of (245057,3)
    data = np.reshape(hsv, (hsv.shape[0], 3))
    predictedLabels = classifier.predict(data)

    # predictedLabels consists of 1 (skin) and 2 (non-skin), needs to change to 0 (non-skin) and 255 (skin)
    # predictedMask = (-(predictedLabels - 1) + 1) * 255

    # resize to match frame shape
    imgLabels = np.zeros(frame.shape, dtype="uint8")
    imgLabels = np.reshape(predictedLabels, (frame.shape[0], frame.shape[1]))
    imgLabels = ((-(imgLabels - 1) + 1) * 255)
    # do bitwsie AND to pull out skin pixels. All skin pixels are anded with 255 and all others are 0
    cv2.imwrite('./resultNew.png', imgLabels)
    imageMask = cv2.imread("./resultNew.png")

    # masks require 1 channel, not 3, so change from BGR to GRAYSCALE
    imgLabels = cv2.cvtColor(imageMask, cv2.COLOR_BGR2GRAY)
    skin = cv2.bitwise_and(frame, frame, mask=imgLabels)

    return imgLabels

# first let's train the data
data, labels = ReadData()
classifier = TrainTree(data, labels)

# next capture the video feed
camera = cv2.VideoCapture(0)

while True:
    ret,frame = camera.read()
    frame = cv2.bilateralFilter(frame,9,75,75)

    if not ret:
        break

    skin = detectSkin(frame)

    # YCrCb values taken from paper Face Segmentation Using Skin-Color Map in Videophone Applications by Chai and Ngan
    im_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))
    skin_ycrcb = cv2.inRange(im_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    # Method 1: Use YCRCB color range to detect skin
    #img,contours,hier = cv2.findContours(skin_ycrcb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Method 2: Use a trained classifier to detect skin color
    img, contours, hier = cv2.findContours(skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # start with maxArea as 0 and update as we find larger contour areas
    maxArea = 0
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            ci = i

    # therefore max contour is contours[ci]
    if ci != 0:
        maxCnt = contours[ci]
    else:
        continue
    cv2.drawContours(frame, maxCnt, -1, (0, 0, 0), 3)
    # now we can find the convex hull of the largest contour
    hull = cv2.convexHull(maxCnt)

    # then find the convex defects
    # defects is an array of format: [start point, end point, farthest point, approximate distance to farthest point]
    hullDefects = cv2.convexHull(maxCnt,returnPoints=False)
    defects = cv2.convexityDefects(maxCnt,hullDefects)

    # now calculate the center of mass of the hand (center of outlined hand) through image moments
    # image moments are a weighted average (moment) of the image pixel' intensities
    M = cv2.moments(maxCnt)
    if M['m00'] != 0:
        # centroid in x is M10/M00 and centroid in y is M01/M00
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    centerMass = (cx, cy)

    # White dot to show center of outlined hand
    cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Center', (cx,cy), font, 0.5, (255, 255, 255), 2)
    output = str(cx) + "    " + str(cy)
    #cv2.putText(frame, output, (cx, cy), font, 0.5, (255, 255, 255), 2)

    farthestDefect = []
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        # start, end, and far are (x,y) coordinates
        start = tuple(maxCnt[s][0])
        end = tuple(maxCnt[e][0])
        far = tuple(maxCnt[f][0])

        # Method 1 to counting finger: count number of defects in between fingers
        # Problem with this method is that it considers defects between a certain range. This range will vary from hand
        # to hand so the range needs to be updated accordingly

        # Check that points are above center of mass and defect point falls within correct range
        if d > 30000 and d < 78000 and (centerMass[1]-far[1] > 20 or (far[0]-centerMass[0] > 50 and abs(centerMass[1]-far[1] < 50))):
            # let's keep track of all the distances to the farthest defect
            farthestDefect.append(far)
            # draw a line joining start point and end point, then draw a circle at the farthest point
            cv2.line(frame, start, end, [0, 255, 0], 1)     # Green line
            cv2.circle(frame, far, 10, [255, 0, 0], 3)      # Blue circle
            count += 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(frame, str(d), far, font, 0.5, (255, 255, 255), 2)
            #cv2.putText(frame, str(start), (start[0]+20,start[1]+20), font, 0.5, (255, 255, 255), 2)
            #cv2.putText(frame, str(end), end, font, 0.5, (0, 0, 0), 2)
            #cv2.putText(frame, str(far), far, font, 0.5, (0, 0, 255), 2)

        '''
        # Method 2 for counting fingers: calculate angle between between
        lenA = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)  # opp side of angle (dist b/w fingers)
        lenB = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)  # left side of angle to be found (len of finger)
        lenC = math.sqrt((start[0] - far[0]) ** 2 + (start[1] - far[1]) ** 2)  # right side of angle to be found (len of finger)

        angA = math.acos((lenB ** 2 + lenC ** 2 - lenA ** 2) / (2 * lenB * lenC))
        angADeg = math.degrees(angA)
        angADeg = round(angADeg,3)

        if (angADeg >= 25 and angADeg <= 80):
            cv2.putText(frame, str(angADeg), (far[0]+20,far[1]+20), font, 0.5, (0, 0, 255), 2)
            # draw a line joining start point and end point, then draw a circle at the farthest point
            cv2.line(frame, start, end, [0, 255, 0], 1)  # Green line
            cv2.circle(frame, far, 10, [255, 0, 0], 3)  # Blue circle
            count += 1
        '''


    font = cv2.FONT_HERSHEY_SIMPLEX
    numFingers = count + 1
    output = "Number of Fingers: " + str(numFingers)
    cv2.putText(frame, output, (100,100), font, 1.0, (255, 255, 255), 2)


    # SEPARATE ATTEMPT TO COUNT FINGERS
    '''
    # now calculate the distance from each defect to the center of the hand
    # distDefectToCenter = []
    distDefectToCenterDict = {}
    distDefectToCenterList = []
    xDefect = []
    yDefect = []
    for i in range(len(farthestDefect)):
        # print("++++++++++++++++++\n{}\n\n".format(farthestDefect[i]))
        x = np.array(farthestDefect[i])
        centerMass = np.array(centerMass)
        # let's first run the point polygon test which determines if a defect point is inside, on,
        # or outside the contour. We want points that are on the contour
        dist = cv2.pointPolygonTest(maxCnt, (x[0], x[1]), False)
        # check if point is above center of mass, or point is on the thumb so to the right and
        # almost level with center of mass
        if (centerMass[1]-x[1] > 80 or (x[0]-centerMass[0] > 100 and abs(centerMass[1]-x[1] < 30))) and dist == 0:
            # then we are looking at defects that lie on the contour (potential finger tips) and we are
            # looking at defects above the center of mass, i.e. potential fingers
            # next eliminate defects that are close together so only 1 defect represents an area
            # if within 30 pixels in x direction and 50 pixels in the y direction, store defect point
            if not xDefect:
                dist = np.sqrt(np.power(x[0] - centerMass[0], 2) + np.power(x[1] - centerMass[1], 2))
                distDefectToCenterList.append(int(dist))
                distDefectToCenterDict[int(dist)] = [x[0], x[1]]
                xDefect.append(x[0])
                yDefect.append(x[1])
            else:
                flagX = 0
                flagY = 0
                for xVal in range(len(xDefect)):
                    if abs(xDefect[xVal] - x[0]) > 20:
                        flagX = 1
                for yVal in range(len(yDefect)):
                    if abs(yDefect[yVal] - x[1]) > 30:
                        flagY = 1
                if flagX == 1 and flagY == 1:
                    # then both x and y values are farther from all defects by at least 30 in x and 50 in y
                    # let's store the values
                    dist = np.sqrt(np.power(x[0] - centerMass[0], 2) + np.power(x[1] - centerMass[1], 2))
                    distDefectToCenterList.append(int(dist))
                    distDefectToCenterDict[int(dist)] = [x[0], x[1]]
                    dist = cv2.pointPolygonTest(maxCnt, (x[0], x[1]), False)

    distDefectToCenterList.sort(reverse=True)
    # print("dictionary: {}\n\n".format(distDefectToCenterList))

    # Red circles represent all defects above the center of mass
    i = 0
#    for key in distDefectToCenterDict:
#        cv2.circle(frame, (distDefectToCenterDict[key][0], distDefectToCenterDict[key][1]), 10, [0, 0, 255], 3)  # Red


    if len(distDefectToCenterList) >= 5:
        print(distDefectToCenterList)
        print(distDefectToCenterList[0])
        print(distDefectToCenterList[1])
        print(distDefectToCenterDict[distDefectToCenterList[0]])
        print(distDefectToCenterDict[distDefectToCenterList[1]])

    # now we can take the top 10 highest distances, and see which of these values fall within a certain
    # angle range. If they fall within an angle range, assume these are fingers
    # let's first assume that if we have multiple defects within a 40 pixel area, they represent one area
    # for no fingers, need to have a threshold that the max distance is at least a certain numbers from
    # the center of mass, if it doesn't meet the threshold, then assume no fingers
    # also assume that coordinates of finger tips should be above the center of mass position
    # maxX = distDefectToCenterDict[distDefectToCenterList[0]][0]
    # maxY = distDefectToCenterDict[distDefectToCenterList[0]][1]
    # cv2.circle(frame, (maxX, maxY), 10, [0, 0, 255], 3)  # Red circle
    # if distDefectToCenterDict[distDefectToCenterList[0]][1] < 50:
    #    print("0 Fingers!!!!")
    '''


    cv2.imshow("contours", frame)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the video capture
camera.release()
cv2.destroyAllWindows()

