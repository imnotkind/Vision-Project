import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg


class Segmentation:
    def __init__(self, modelname='deeplab_resnet152_voc', palletename='pascal_voc'):
        self.ctx = mx.cpu(0)
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
        img = image.imread(filename + '.png')
        img = self.transform_fn(img)
        img = img.expand_dims(0).as_in_context(self.ctx)
        output = self.model.demo(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        mask = get_color_pallete(predict, self.palletename)
        mask.save(filename + '_label.png')


    def show_img(self, filename):
        tmp = mpimg.imread(filename)
        plt.imshow(tmp)
        plt.show()




seg = Segmentation('deeplab_resnet152_voc', 'pascal_voc')
exit(0)

filelist = ['255', '2740', '2877', '2921', '3097']

filelist = list(map(lambda x : 'sample/'+x, filelist))

for i in filelist:
    seg.process(i)