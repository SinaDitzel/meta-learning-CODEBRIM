import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from lib.Models.network import spatial_pyramid_pooling


class CODEBRIMetaQNN1(nn.Module):
    def __init__(self,num_classes=6):
        super(CODEBRIMetaQNN1, self).__init__()
        #in 3,224,224
        bn_val = 1e-4
        feature_extractor_list =[]
        feature_extractor_list.append(("conv1", nn.Conv2d(3, 256 , 9,2, bias = False))) #out 256,109
        feature_extractor_list.append(("batchnorm1", nn.BatchNorm2d(num_features=256, eps=bn_val)))
        feature_extractor_list.append(('relu1' , nn.ReLU(inplace=True)))

        feature_extractor_list.append(("conv2", nn.Conv2d(256, 32 , 3,1, bias = False))) #out 256,109
        feature_extractor_list.append(("batchnorm2", nn.BatchNorm2d(num_features=32, eps=bn_val)))
        feature_extractor_list.append(('relu2' , nn.ReLU(inplace=True)))

        feature_extractor_list.append(("conv3", nn.Conv2d(32, 256 , 5,1,bias = False))) #out 256,109
        feature_extractor_list.append(("batchnorm3", nn.BatchNorm2d(num_features=256, eps=bn_val)))
        feature_extractor_list.append(('relu3' , nn.ReLU(inplace=True)))
        
        feature_extractor_list.append(("conv4", nn.Conv2d(256,256,7,2,bias = False)))#out 256,109
        feature_extractor_list.append(("batchnorm4",nn.BatchNorm2d(num_features=256, eps=bn_val)))
        feature_extractor_list.append(('relu4' , nn.ReLU(inplace=True)))
        
        classifier_list =[]
        classifier_list.append(("fc1",nn.Linear(7680,128,bias= False)))
        classifier_list.append(("batchnorm_fc1",  nn.BatchNorm1d(num_features=128,eps=bn_val)))

        classifier_list.append(("dropout", nn.Dropout(p=0.5)))
        classifier_list.append(("fc2", nn.Linear(128, num_classes, bias=False)))
        self.feature_extractor = nn.Sequential(collections.OrderedDict(feature_extractor_list))
        self.classifier = nn.Sequential(collections.OrderedDict(classifier_list))


    def forward(self, x):
        x = self.feature_extractor(x)
        x = spatial_pyramid_pooling(x,4)
        x = torch.sigmoid(self.classifier(x))
        return x


class VGG(nn.Module): #VGG11
    def __init__(self, feature_config, classifier_config, num_classes,net_input, bn_val,do_drop):
        super(VGG, self).__init__()
        _,in_c, in_w, in_h = net_input.shape
        feature_extractor_list =[]
        #in 3,224,224
        for x in feature_config:#out width:112-56-28-14,7
            if x == 'mp':
                feature_extractor_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                in_c, in_w, in_h = in_c, in_w//2, in_h//2
            else:
                feature_extractor_list.append(nn.Conv2d(in_c, x, kernel_size=3, stride=1, padding=1, bias=False))
                if bn_val > 0.0:
                    feature_extractor_list.append(nn.BatchNorm2d(x, eps = bn_val))
                feature_extractor_list.append(nn.ReLU(inplace=True))
                in_c, in_w, in_h = x, in_w, in_h
        
        classifier_list = []
        
        in_features = in_c* in_w* in_h
        for x in classifier_config:
            classifier_list.append(nn.Linear(in_features,x,bias = False))
            if bn_val > 0.0:
                classifier_list.append(nn.BatchNorm1d(x, eps = bn_val))
            classifier_list.append(nn.ReLU(inplace=True))
            in_features = x
        if do_drop != 0:
            classifier_list.append(nn.Dropout(p = do_drop))
        classifier_list.append(nn.Linear(in_features,num_classes))

        self.feature_extractor = nn.Sequential(*feature_extractor_list)
        self.classifier = nn.Sequential(*classifier_list)
        self.gpu_mem_req = 0

    def forward(self,x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


VGG_A_feature_config =  [64, 'mp', 128, 'mp',256, 256, 'mp',512,512,'mp',512,512,'mp']
VGG_A_classifier_config =  [4096,4096] 
class VGG_A(VGG): #VGG11
    def __init__(self, num_classes, net_input, bn_vl, do_drop):
        super(VGG_A, self).__init__(VGG_A_feature_config, VGG_A_classifier_config, num_classes,net_input, bn_vl,do_drop)

VGG_D_feature_config =  [64, 64, 'mp', 128, 128, 'mp',256, 256, 256, 256, 'mp',512,512,512,512,'mp',512,512,512,512,'mp']
VGG_D_classifier_config = [4096,4096]
class VGG_D(VGG): #VGG11
    def __init__(self, num_classes, net_input, bn_vl, do_drop):
        super(VGG_D, self).__init__(VGG_D_feature_config, VGG_D_classifier_config, num_classes,net_input, bn_vl,do_drop)


class AlexNet(nn.Module): #VGG11
    def __init__(self, num_classes,net_input, bn_val,do_drop):
        super(AlexNet, self).__init__()
        feature_extractor_list =[]
        feature_extractor_list.append(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False))
        if bn_val > 0.0:
            feature_extractor_list.append(nn.BatchNorm2d(x, eps = bn_val))
        feature_extractor_list.append(nn.ReLU(inplace=True))
        feature_extractor_list.append(nn.MaxPool2d(kernel_size=3, stride=2))
        feature_extractor_list.append(nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False))
        if bn_val > 0.0:
            feature_extractor_list.append(nn.BatchNorm2d(x, eps = bn_val))
        feature_extractor_list.append(nn.ReLU(inplace=True))
        feature_extractor_list.append(nn.MaxPool2d(kernel_size=3, stride=2))
        feature_extractor_list.append(nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False))
        if bn_val > 0.0:
            feature_extractor_list.append(nn.BatchNorm2d(x, eps = bn_val))
        feature_extractor_list.append(nn.ReLU(inplace=True))
        feature_extractor_list.append(nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False))
        if bn_val > 0.0:
            feature_extractor_list.append(nn.BatchNorm2d(x, eps = bn_val))
        feature_extractor_list.append(nn.ReLU(inplace=True))
        feature_extractor_list.append(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        if bn_val > 0.0:
            feature_extractor_list.append(nn.BatchNorm2d(x, eps = bn_val))
        feature_extractor_list.append(nn.ReLU(inplace=True))
        feature_extractor_list.append(nn.MaxPool2d(kernel_size=3, stride=2))

        classifier_list = []
        classifier_list.append(nn.Linear(in_features,num_classes))

        self.feature_extractor = nn.Sequential(*feature_extractor_list)
        self.classifier = nn.Sequential(*classifier_list)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return torch.sigmoid(x)

#class Alexnet
#class T-CNN
#class Densenet-121
#class WRN_28_4
