from torch import nn
from torch.nn import functional as F


class CBL(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding=0, stride=1):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, X):
        return F.leaky_relu(self.bn(self.conv(X)))


class ResUnit(nn.Module):
    def __init__(self, in_c):
        super(ResUnit, self).__init__()
        self.cbl1 = CBL(in_c=in_c, out_c=in_c // 2, kernel_size=1)
        self.cbl2 = CBL(in_c=in_c // 2, out_c=in_c, kernel_size=3, padding=1)

    def forward(self, X):
        return self.cbl2(self.cbl1(X)) + X


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.cbl = CBL(in_c=3, out_c=32, padding=1, kernel_size=3)

        self.res1 = res_block(res_num=1, in_c=32, out_c=64, stride=2)
        self.res2 = res_block(res_num=2, in_c=64, out_c=128, stride=2)
        self.res3 = res_block(res_num=8, in_c=128, out_c=256, stride=2)
        self.res4 = res_block(res_num=8, in_c=256, out_c=512, stride=2)
        self.res5 = res_block(res_num=4, in_c=512, out_c=1024, stride=2)

    def forward(self, X):
        X = self.cbl(X)
        out3 = self.res3(self.res2(self.res1(X)))
        out2 = self.res4(out3)
        out1 = self.res5(out2)
        return out1, out2, out3


class Yolov3(nn.Module):
    def __init__(self):
        super(Yolov3, self).__init__()
        self.darknet = DarkNet53()
        self.yolo1 = create_yolo_block(1024, 512)
        self.yolo2 = create_yolo_block(768, 256)
        self.yolo3 = create_yolo_block(384, 128)

        self.up_sample1 = create_up_sample(512, 256)
        self.up_sample2 = create_up_sample(256, 128)

        self.cbl1 = CBL(512, 1024, kernel_size=3, padding=1)
        self.cbl2 = CBL(256, 512, kernel_size=3, padding=1)
        self.cbl3 = CBL(128, 256, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(1024, 255, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 255, kernel_size=1)
        self.conv3 = nn.Conv2d(256, 255, kernel_size=1)

    def forward(self, X):
        o1, o2, o3 = self.darknet(X)
        y1 = self.yolo1(o1)
        out1 = self.conv1(self.cbl1(y1))

        o2 = torch.cat((o2, self.up_sample1(y1)), dim=1)
        y2 = self.yolo2(o2)
        out2 = self.conv2(self.cbl2(y2))

        o3 = torch.cat((o3, self.up_sample2(y2)), dim=1)
        out3 = self.conv3(self.cbl3(self.yolo3(o3)))
        return out1, out2, out3


def create_up_sample(in_c, out_c):
    layer_list = [CBL(in_c=in_c, out_c=out_c, kernel_size=1),
                  nn.UpsamplingNearest2d(scale_factor=2)]
    return nn.Sequential(*layer_list)


def create_yolo_block(in_c, out_c):
    layer_list = []
    for _ in range(2):
        layer_list.append(CBL(in_c=in_c, out_c=out_c, kernel_size=1))
        layer_list.append(CBL(in_c=out_c, out_c=in_c, kernel_size=3, padding=1))
    layer_list.append(CBL(in_c=in_c, out_c=out_c, kernel_size=1))
    return nn.Sequential(*layer_list)


def res_block(res_num, in_c, out_c, stride=1):
    res_list = [CBL(in_c=in_c, out_c=out_c, kernel_size=3, padding=1, stride=stride)]
    for idx in range(res_num):
        res_list.append(ResUnit(out_c))
    return nn.Sequential(*res_list)


# if __name__ == '__main__':
#     import torch
#
#     test = torch.randn(size=(1, 3, 416, 416), requires_grad=False)
#     net = Yolov3()
#     o1, o2, o3 = net(test)
#     print(o1.shape, o2.shape, o3.shape, sep='\n')
