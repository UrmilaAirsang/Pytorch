import torch
import torch.nn as nn

import numpy as np


yolo_default_cfg = {'type': 'yolo', 'mask': [[0, 1, 2],[3, 4, 5], [6, 7, 8]],
                    'anchors': np.array([[ 10.,  13.],
       [ 16.,  30.],
       [ 33.,  23.],
       [ 30.,  61.],
       [ 62.,  45.],
       [ 59., 119.],
       [116.,  90.],
       [156., 198.],
       [373., 326.]]), 'classes': 4, 'num': 9, 'jitter': '.3', 'ignore_thresh': '.7',
                    'truth_thresh': 1, 'random': 1, 'stride': [32, 16, 8]}


def _make_encoder(features, use_pretrained):
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features)
    plane_segmentation_decode = _make_scratch([256, 512, 1024, 2048], features)

    return pretrained, scratch, plane_segmentation_decode


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


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
    )
    return scratch


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class ResidualConvUnitBN(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.bn = nn.BatchNorm2d(features)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(self.bn(out))
        out = self.bn(self.conv2(out))

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters, strict=False)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride
        # self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):

        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        # print("p: ", p.shape)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        else:  # inference
            self.create_grids((nx, ny), p.device)
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class CustomNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True, yolo_cfg=yolo_default_cfg, nc=4, plane_classes=8):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(CustomNet, self).__init__()

        use_pretrained = False if path is None else True
        self.nc = nc

        self.pretrained, self.scratch, self.plane_segmentation_decode = _make_encoder(features, use_pretrained)
        self.yolo_cfg = yolo_cfg

        # YOLOv3
        self.yolo_init = nn.Conv2d(2048, 169, kernel_size=3, stride=1, padding=1, bias=True)
        self.yolo_init2 = nn.Conv2d(169, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.yolo_preconv1 = ResidualConvUnitBN(features)
        self.yolo_preconv2 = ResidualConvUnitBN(features)
        self.yolo_postconv = nn.Conv2d(features, 27, kernel_size=3, stride=1, padding=1, bias=True)
        self.yolo_head1 = YOLOLayer(anchors=yolo_cfg['anchors'][yolo_cfg['mask'][0]], nc=yolo_cfg['classes'],
                              stride=yolo_cfg['stride'][0])
        self.yolo_mid_block = nn.Sequential(
                                        nn.Conv2d(27, features, kernel_size=3, stride=1, padding=1, bias=True),
                                        nn.BatchNorm2d(features),
                                        ResidualConvUnitBN(features),
                                        ResidualConvUnitBN(features),
                                        ResidualConvUnitBN(features),
                                        ResidualConvUnitBN(features),
                                        nn.Conv2d(features, 27, kernel_size=3, stride=1, padding=1, bias=True),
                                            )
        self.yolo_head2 = YOLOLayer(anchors=yolo_cfg['anchors'][yolo_cfg['mask'][1]], nc=yolo_cfg['classes'],
                                    stride=yolo_cfg['stride'][1])
        self.yolo_end_block = nn.Sequential(
                                        nn.Conv2d(27, features, kernel_size=3, stride=1, padding=1, bias=True),
                                        nn.BatchNorm2d(features),
                                        ResidualConvUnitBN(features),
                                        ResidualConvUnitBN(features),
                                        ResidualConvUnitBN(features),
                                        ResidualConvUnitBN(features),
                                        nn.Conv2d(features, 27, kernel_size=3, stride=1, padding=1, bias=True),
                                            )
        self.yolo_head3 = YOLOLayer(anchors=yolo_cfg['anchors'][yolo_cfg['mask'][2]], nc=yolo_cfg['classes'],
                                    stride=yolo_cfg['stride'][2])
        self.yolo_heads = [self.yolo_head1, self.yolo_head2, self.yolo_head3]


        # MiDaS
        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        # Plane Segmentation
        self.plane_segmentation_decode.refinenet4 = FeatureFusionBlock(features)
        self.plane_segmentation_decode.refinenet3 = FeatureFusionBlock(features)
        self.plane_segmentation_decode.refinenet2 = FeatureFusionBlock(features)
        self.plane_segmentation_decode.refinenet1 = FeatureFusionBlock(features)

        self.plane_segmentation_decode.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, plane_classes, kernel_size=1, stride=1, padding=0),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)
        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        # print("layer4: ", layer_4.shape)

        # MiDas
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        midas_out = torch.squeeze(out, dim=1)

        # Plane Segmentation

        seg_layer_1_rn = self.plane_segmentation_decode.layer1_rn(layer_1)
        seg_layer_2_rn = self.plane_segmentation_decode.layer2_rn(layer_2)
        seg_layer_3_rn = self.plane_segmentation_decode.layer3_rn(layer_3)
        seg_layer_4_rn = self.plane_segmentation_decode.layer4_rn(layer_4)

        seg_path_4 = self.plane_segmentation_decode.refinenet4(seg_layer_4_rn)
        seg_path_3 = self.plane_segmentation_decode.refinenet3(seg_path_4, seg_layer_3_rn)
        seg_path_2 = self.plane_segmentation_decode.refinenet2(seg_path_3, seg_layer_2_rn)
        seg_path_1 = self.plane_segmentation_decode.refinenet1(seg_path_2, seg_layer_1_rn)

        plane_pred = self.plane_segmentation_decode.output_conv(seg_path_1)

        # YOLOv3 Out
        yolo_out = []
        yolo_1 = self.yolo_init(layer_4)
        # yolo_1 = yolo_1.view(yolo_1.shape[0], -1, 13, 13)
        yolo_1b = self.yolo_init2(yolo_1)
        yolo_2 = self.yolo_preconv1(yolo_1b)
        yolo_3 = self.yolo_preconv2(yolo_2)
        yolo_post = self.yolo_postconv(yolo_3)
        yolo_out.append(self.yolo_head1(yolo_post))
        yolo_mid = self.yolo_mid_block(yolo_post)
        yolo_out.append(self.yolo_head2(yolo_mid))
        yolo_end = self.yolo_end_block(yolo_mid)
        yolo_out.append(self.yolo_head3(yolo_end))


        if self.training:
            return midas_out, plane_pred, yolo_out
        else:
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return midas_out, plane_pred, x, p