import numpy as np
import cv2
from numpy.linalg import lstsq

# ============================================================================

FINAL_LINE_COLOR = (0, 255, 0)
WORKING_LINE_COLOR = (255, 0, 0)

# ============================================================================


class Preprocessing(object):
    def __init__(self):
        self.ori_img = None

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.maskBox = []
        self.clickCount = 0
    
    def perpendicular(self, a) :
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        b = np.array(b)
        return b/np.linalg.norm(b)

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
            if (self.clickCount == 2):
                v_first = [self.points[1][0] - self.points[0][0], self.points[1][1] - self.points[0][1]]
                v_first_perp = self.perpendicular(v_first)
                u = [self.current[0] - self.points[1][0], self.current[1] - self.points[1][1]]
                self.points[2] = tuple([int(num) for num in [self.points[1][0], self.points[1][1]] 
                                    + (v_first_perp*np.dot(u, v_first_perp)/np.dot(v_first_perp, v_first_perp))])
                self.points[3] = tuple([int(num) for num in np.subtract([self.points[2][0], self.points[2][1]], v_first)])
                self.temp = tuple(u)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            # print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            # self.points.append((x, y))
            if (self.clickCount == 0):
                self.points.append((x, y))
                self.clickCount += 1
            elif (self.clickCount == 1):
                self.points.append((x, y))
                self.clickCount += 1
                self.points.append((0, 0))
                self.points.append((0, 0))
            elif (self.clickCount == 2):
                self.clickCount = 0
                self.maskBox.append(self.points)
                self.points = []
            

    def addMask(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow("Editing MaskBox", flags=cv2.WINDOW_AUTOSIZE)
        clone = self.ori_img.copy()
        cv2.waitKey(1)
        cv2.setMouseCallback("Editing MaskBox", self.on_mouse)

        if (len(self.maskBox) > 0):
            for box in self.maskBox:
                cv2.polylines(clone, np.array([box]), True, FINAL_LINE_COLOR, 1)

        while(True):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # print("maskBox: ", self.maskBox)
            
            if (len(self.points) > 0):
                clone = self.ori_img.copy()
                if (len(self.maskBox) > 0):
                    for box in self.maskBox:
                        cv2.polylines(clone, np.array([box]), True, FINAL_LINE_COLOR, 1)    

                if (self.clickCount < 2):
                    cv2.line(clone, self.points[-1], self.current, WORKING_LINE_COLOR)
                else:
                    # Draw all the current polygon segments
                    cv2.polylines(clone, np.array([self.points]), True, WORKING_LINE_COLOR, 1)

            # Update the window
            cv2.imshow("Editing MaskBox", clone)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            # if cv2.waitKey(50) == 27: # ESC hit
            #     self.done = True

            key = cv2.waitKey(1)
            if (key == ord("r")):
                clone = self.ori_img.copy()
                if (len(self.maskBox) > 0):
                    self.maskBox.pop()
                
                for box in self.maskBox:
                    cv2.polylines(clone, np.array([box]), True, FINAL_LINE_COLOR, 1)
            
            elif (key == ord("c")):
                break
            elif (key == -1):
                continue

        cv2.destroyWindow("Editing MaskBox")
    
    def removeMask(self):
        cv2.namedWindow("Removing MaskBox", flags=cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        active = None

        while (True):
            
            clone = self.ori_img.copy()

            if (len(self.maskBox) > 0) or not (active == None):
                
                for box in self.maskBox:
                    cv2.polylines(clone, np.array([box]), True, FINAL_LINE_COLOR, 1)
                
                if not (active == None):
                    cv2.polylines(clone, np.array([active]), True, WORKING_LINE_COLOR, 1)

                key = cv2.waitKey(1)

                if (key == ord("s")):
                    if (active == None):
                        active = self.maskBox.pop(0)
                    else:
                        self.maskBox.append(active)
                        active = self.maskBox.pop(0)

                elif (key == ord("r")) and not (active == None):
                    active = None
                elif (key == ord("c")):
                    if not (active == None):
                        self.maskBox.append(active)
                    break
            else:
                print("MaskBox is empty")
                break
            
            cv2.imshow("Removing MaskBox", clone)
            # print("Mask: ", self.maskBox)
        
        cv2.destroyWindow("Removing MaskBox")

    def getCrop(self):
        crop = []
        if (len(self.maskBox) > 0):
            for box in self.maskBox:

                cnt = np.array(box)
                rect = cv2.minAreaRect(cnt)
                # print("rect: {}".format(rect))

                # the order of the box points: bottom left, top left, top right,
                # bottom right
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # print("bounding box: {}".format(box))
                # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                # get width and height of the detected rectangle
                width = int(rect[1][0])
                height = int(rect[1][1])

                src_pts = box.astype("float32")
                # coordinate of the points in box points after the rectangle has been
                # straightened
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # directly warp the rotated rectangle to get the straightened rectangle
                warped = cv2.warpPerspective(self.ori_img.copy(), M, (width, height))
                crop.append(warped)
        
        return crop
    
    def setCloudMask(self, message):

        self.maskBox = []

        if message["path"] == '/':

            # messageSorted = sorted(message["data"])
            for value in message["data"].values():
                mask = []
                points = value.split(';')
                for pt in points:
                    point = pt.split(',')
                    mask.append((int(point[0]), int(point[1])))
                
                self.maskBox.append(mask)
        
        # print(self.maskBox)

    def getCloudMask(self):
        mask = {}

        for idx, box in enumerate(self.maskBox):
            mask['slot_'+str(idx)] = str(box[0][0])+','+str(box[0][1])+';'+str(box[1][0])+','+str(box[1][1])+';'+str(box[2][0])+','+str(box[2][1])+';'+str(box[3][0])+','+str(box[3][1])
        
        return mask

    def getMask(self):
        return self.maskBox
    
    def setMask(self, mask):
        self.maskBox = mask

    def setImage(self, img):
        self.ori_img = img


# if __name__ == "__main__":
#     image = cv2.imread('test.jpg')
#     image = cv2.resize(image, (1024, 680))
#     prepro = Preprocessing()
#     prepro.setImage(image)
#     prepro.addMask()
#     prepro.removeMask()
#     img_crop = prepro.getCrop()
#     for idx, img in enumerate(img_crop):
#         cv2.imshow(str(idx), img)
    
#     cv2.waitKey()
#     cv2.destroyAllWindows()