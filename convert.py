import cv2
import os
import sys
from fpdf import FPDF
import numpy as np
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
from PIL import Image

class Segmentation:
    def __init__(self, modelname='deeplab_resnet152_voc', palletename='pascal_voc'):
        self.ctx = mx.gpu(0)
        self.model = gluoncv.model_zoo.get_model(modelname, pretrained=True, root='.mxnet/models', ctx=self.ctx)

        self.transform_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        if '_voc' in modelname:
            self.palletename = 'pascal_voc'
        elif '_ade' in modelname:
            self.palletename = 'ade20k'
        else:
            print("invalid model name")
            exit(0)

    def process(self, filename):
        origimg = image.imread(filename + '.png')
        img = self.transform_fn(origimg)
        img = img.expand_dims(0).as_in_context(self.ctx)
        output = self.model.demo(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

        s = set([])
        for (x,y), value in np.ndenumerate(predict):
            s.add(value)
        print(s) #0:background, 15:person

        mask = get_color_pallete(predict, self.palletename)
        mask.save(filename + '_mask.png')

        mask2 = np.array(mask.convert('RGB'))

        overlay = cv2.addWeighted(origimg.asnumpy(),0.5,mask2,0.5,0)
        overlay = Image.fromarray(overlay)
        overlay.save(filename + '_overlay.png')



    def show_img(self, filename):
        tmp = mpimg.imread(filename)
        plt.imshow(tmp)
        plt.show()

def level1(video):
    vidcap = cv2.VideoCapture(video)
    totalframe = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("TOTAL FRAME COUNT : ",totalframe) 
    print("FPS : ", vidcap.get(cv2.CAP_PROP_FPS))


    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("WIDTH : ", width)
    print("HEIGHT : ", height)

    ##################################### calculate transition
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
            dist = 1

            if count != 0:
                white = np.zeros((height, width, 3), np.uint8)
                white[:] = (255, 255, 255)
                dist = cv2.norm(image, white, cv2.NORM_L2) #white-like measure
                

                if predist + 1000 < dist:
                    if fadestat < -6:
                        fadeend = count - 1
                        transition.append(("down", fadestart, fadeend))
                        fadestart = count - 1
                        fadestat = 0
                    elif fadestat < 0:
                        fadestart = count - 1
                        fadestat = 0
                    else:
                        fadestat += 1
                elif predist - 1000 > dist:
                    if fadestat > 6:
                        fadeend = count - 1
                        transition.append(("up", fadestart, fadeend))
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
                        transition.append(("down", fadestart, fadeend))
                    elif fadestat > 6:
                        fadeend = count - 1
                        transition.append(("up", fadestart, fadeend))
                        
                    fadestart = count - 1
                    fadestat = 0
                #print(dist / predist, count, fadestat)


            count += 1
            preimage = image
            predist = dist
        else:
            break


    transition2 = []
    pre = None
    for i in transition:
        print(i)
        if(pre != None and pre[0] == "down" and i[0] == "up"):
            transition2.append((pre[1], i[2]))
        pre = i


    transition3 = []
    pre = 0
    for (i,j) in transition2:
        transition3.append((pre, i-1))
        pre = j
    transition3.append((pre, totalframe-1))

    for j in transition3:
        print(j)
    
    ########################################## overlay

    for (i,j) in transition3:

    

     
    vidcap.release()
    cv2.destroyAllWindows()

    

def pic2pdf(width, height, count):
    pdf = FPDF('L','mm',(height,width))
    for i in range(count):
        pdf.add_page()
        pdf.image(tmp_dir + str(i) + '.png', 0, 0, width, height)

    pdf.output("./output.pdf","F")

level1("./video/level1/1.mp4")