import cv2
import os
import sys
from fpdf import FPDF

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
    while vidcap.isOpened():
        success, image = vidcap.read()

        if success:
            #cv2.imwrite(tmp_dir + str(count) +'.png', image)
            if preimage != None:
                #diffimage = cv2.absdiff(preimage, image)
                #diffimage = cv2.cvtColor(diffimage, cv2.COLOR_BGR2GRAY)
                dist = norm(image, preimage, NORM_L2)
                print(dist, count)
            count += 1
            preimage = image
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