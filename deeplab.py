import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
import numpy as np
import cv2
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

        marginmap = np.zeros((height, width), np.uint8)
        for (h,w), value in np.ndenumerate(predict):
            if marginmap[h][w] == 0 and value == 15:
                for u in range(-20,20):
                    for v in range(-20,20):
                        try:
                            marginmap[h+u][w+v] = 1
                            predict[h+u][w+v] = 15.0
                        except:
                            pass

        mask = get_color_pallete(predict, self.palletename)
        mask.save(filename + '_marginmask.png')

        mask2 = np.array(mask.convert('RGB'))

        overlay = cv2.addWeighted(origimg.asnumpy(),0.5,mask2,0.5,0)
        overlay = Image.fromarray(overlay)
        overlay.save(filename + '_marginoverlay.png')



    def show_img(self, filename):
        tmp = mpimg.imread(filename)
        plt.imshow(tmp)
        plt.show()




seg = Segmentation('deeplab_resnet152_voc', 'pascal_voc')

filelist = ['255', '2740', '2877', '2921', '3097']

filelist = list(map(lambda x : 'sample/'+x, filelist))

for filename in filelist:
    
    seg.process(filename)
