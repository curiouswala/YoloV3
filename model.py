import torch
import torch.nn as nn

import yolo_decoder as yolo_model

def _make_resnet_encoder(use_pretrained):
	pretrained = _make_pretrained_resnext101_wsl(use_pretrained)

	return pretrained

def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
	resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
	return _make_resnet_backbone(resnet)

class Model_Head(nn.Module):
    def __init__(self, cfg ,non_negative=True):

        super(Model_Head, self).__init__()

        # self.pretrained, self.scratch= _make_encoder(backbone="resnext101_wsl", features=256, use_pretrained=True)
        # self.scratch.refinenet4 = FeatureFusionBlock(features)
        # self.scratch.refinenet3 = FeatureFusionBlock(features)
        # self.scratch.refinenet2 = FeatureFusionBlock(features)
        # self.scratch.refinenet1 = FeatureFusionBlock(features)
        # self.scratch.output_conv = nn.Sequential(
        #     nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear"),
        #     nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True) if non_negative else nn.Identity(),
        # )
        self.encoder = _make_resnet_encoder(True)
        self.yolo_conv1 = nn.Conv2d( 2048, 1024, 1, padding= 0)
        self.yolo_conv2 = nn.Conv2d( 1024, 512, 1, padding= 0)
        self.yolo_conv3 = nn.Conv2d( 512, 256, 1, padding= 0)
      
        self.yolo_decoder=yolo_model.Darknet(cfg)

    def midas_encoder(self, x):
        ##  encoder head
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        return layer_1, layer_2, layer_3, layer_4


    def midas_decoder(self, layer_1, layer_2, layer_3, layer_4):
        ## Midas_branch
        midas_layer_1_rn = self.scratch.layer1_rn(layer_1)
        midas_layer_2_rn = self.scratch.layer2_rn(layer_2)
        midas_layer_3_rn = self.scratch.layer3_rn(layer_3)
        midas_layer_4_rn = self.scratch.layer4_rn(layer_4)

        midas_path_4 = self.scratch.refinenet4(midas_layer_4_rn)
        midas_path_3 = self.scratch.refinenet3(midas_path_4, midas_layer_3_rn)
        midas_path_2 = self.scratch.refinenet2(midas_path_3, midas_layer_2_rn)
        midas_path_1 = self.scratch.refinenet1(midas_path_2, midas_layer_1_rn)

        midas_out = self.scratch.output_conv(midas_path_1)
        return torch.squeeze(midas_out, dim=1)

    # def yolo_decoder(self,Yolo_36, Yolo_61, Yolo_75):
    #     ## Yolo_branch
    #     # y_1024 = self.yolo_conv2(layer_4)
    #     # y_512 = self.yolo_conv3(layer_3)
    #     # y_256 = self.yolo_conv4(layer_2)
        
    #     yolo_out= self.yolo_decoder_old.forward( Yolo_75, Yolo_61,Yolo_36 )
    #     return yolo_out

    def planercnn_decoder(self,x):
        
        layer_1, layer_2, layer_3, layer_4= self.midas_encoder(x)
        layers= [layer_1, layer_2, layer_3, layer_4]

        planer_out= maskrcnn(x, layers)
        return planer_out

    def forward(self, x):
        layer_1 = self.encoder.layer1(x)
        layer_2 = self.encoder.layer2(layer_1)
        layer_3 = self.encoder.layer3(layer_2)
        layer_4 = self.encoder.layer4(layer_3)

        Yolo_75 = self.yolo_conv1(layer_4)
        Yolo_61 = self.yolo_conv2(layer_3)
        Yolo_36 = self.yolo_conv3(layer_2)

        if not self.training:
          inf_out, train_out = self.yolo_decoder(Yolo_75,Yolo_61,Yolo_36)
          yolo_out=[inf_out, train_out]
        else:
          yolo_out = self.yolo_decoder(Yolo_75,Yolo_61,Yolo_36)

    # #   layer_1, layer_2, layer_3, layer_4 = self.midas_encoder(x)
    #   # midas_out = self.midas_decoder(layer_1, layer_2, layer_3, layer_4)
        # yolo_out = self.yolo_decoder(Yolo_75, Yolo_61,Yolo_36)
    #   # planer_out= self.planercnn_decoder(x)
        return yolo_out

