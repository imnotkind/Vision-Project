import cv2
import os
import sys
from fpdf import FPDF
import numpy as np

tmp_dir = "./tmp/"


def level1(video):
    vidcap = cv2.VideoCapture(video)
    print("TOTAL FRAME COUNT : ",vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print("FPS : ", vidcap.get(cv2.CAP_PROP_FPS))


    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("WIDTH : ", width)
    print("HEIGHT : ", height)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 0 based index frame
    count = 0
    preimage = None
    transition = []
    fadestat = 0
    fadestart = 0
    fadeend = 0

    while vidcap.isOpened():
        success, image = vidcap.read()

        if success:
            #cv2.imwrite(tmp_dir + str(count) +'.png', image)
            dist = 0

            if count != 0:
                white = np.zeros((height, width, 3), np.uint8)
                white[:] = (255, 255, 255)
                dist = cv2.norm(image, white, cv2.NORM_L2)
                

                if predist + 500 < dist:
                    if fadestat < -6:
                        fadeend = count - 1
                        print("FADE DEC", fadestart, fadeend)
                        fadestart = count - 1
                        fadestat = 0
                    elif fadestat < 0:
                        fadestart = count - 1
                        fadestat = 0
                    else:
                        fadestat += 1
                elif predist - 500 > dist:
                    if fadestat > 6:
                        fadeend = count - 1
                        print("FADE ENC", fadestart, fadeend)
                        fadestart = count - 1
                        fadestat = 0
                    elif fadestat > 0:
                        fadestart = count - 1
                        fadestat = 0
                    else:
                        fadestat -= 1
                    
                else:
                    if fadestat < -6:
                        fadeend = count - 1
                        print("FADE DEC", fadestart, fadeend)
                        fadestart = count - 1
                    elif fadestat > 6:
                        fadeend = count - 1
                        print("FADE ENC", fadestart, fadeend)
                        fadestart = count - 1

                    fadestat = 0
                print(dist, count, fadestat)


            count += 1
            preimage = image
            predist = dist
        else:
            break


    vidcap.release()
    cv2.destroyAllWindows()

    

def pic2pdf(width, height, count):
    pdf = FPDF('L','mm',(height,width))
    for i in range(count):
        pdf.add_page()
        pdf.image(tmp_dir + str(i) + '.png', 0, 0, width, height)

    pdf.output("./output.pdf","F")

level1("./video/level1/1.mp4")