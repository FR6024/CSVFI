import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import ResBlock
from cupy_module import dsepconv


def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):
        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class upSplit(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3,3), stride=(1,2,2), padding=1),
                 ]
            )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x, output_size):
        x = self.upconv[0](x, output_size=output_size)
        return x

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)


class UNet_3D_3D(nn.Module):
    def __init__(self, n_inputs=4, joinType="concat", ks=5, dilation=1):
        super().__init__()

        nf = [64, 64, 64, 32]
        split_size = [1, 2, 8, 8]
        nh = [2, 4, 8, 16]
        depth = [2, 2, 6, 2]
        self.joinType = joinType
        self.n_inputs = n_inputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)
        from model.encoder import Encoder
        self.encoder = Encoder(nf, depth=depth, nh=nh, split_size=split_size)

        self.decoder = nn.Sequential(
            upSplit(nf[0], nf[1]),
            upSplit(nf[1]*growth, nf[2]),
            upSplit(nf[2]*growth, nf[3]),
        )

        def SmoothNet(inc, ouc):
            return torch.nn.Sequential(
                Conv_3d(inc, ouc, kernel_size=3, stride=1, padding=1, batchnorm=False),
                ResBlock(ouc, kernel_size=3),
            )

        nf_out = 64
        self.smooth_ll = SmoothNet(nf[1]*growth, nf_out)
        self.smooth_l = SmoothNet(nf[2]*growth, nf_out)
        self.smooth = SmoothNet(nf[3]*growth, nf_out)

        self.predict_ll = SynBlock(n_inputs, nf_out, ks=ks)
        self.predict_l = SynBlock(n_inputs, nf_out, ks=ks)
        self.predict = SynBlock(n_inputs, nf_out, ks=ks)

    def forward(self, frames, edges):
        images = torch.stack(frames, dim=2)
        edges = torch.stack(edges, dim=2)
        _, _, _, H, W = images.shape

        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        images = images - mean_
        edges_ = edges.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        edges = edges - edges_

        x_0, x_1, x_2, x_3, x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4, x_3.size()))
        dx_3 = joinTensors(dx_3 , x_3 , type=self.joinType)

        dx_2 = self.lrelu(self.decoder[1](dx_3, x_2.size()))
        dx_2 = joinTensors(dx_2 , x_2 , type=self.joinType)

        dx_1 = self.lrelu(self.decoder[2](dx_2, x_1.size()))
        dx_1 = joinTensors(dx_1 , x_1 , type=self.joinType)

        fea3 = self.smooth_ll(dx_3)
        fea2 = self.smooth_l(dx_2)
        fea1 = self.smooth(dx_1)

        out_ll = self.predict_ll(fea3, frames, edges, x_2.size()[-2:])  # out shape x_2.size()[-2:]

        out_l = self.predict_l(fea2, frames, edges, x_1.size()[-2:])
        out_l = F.interpolate(out_ll, size=out_l.size()[-2:], mode='bilinear') + out_l

        out = self.predict(fea1, frames, edges, x_0.size()[-2:])
        out = F.interpolate(out_l, size=out.size()[-2:], mode='bilinear') + out

        if self.training:
            return out_ll, out_l, out
        else:
            return out


class MySequential(nn.Sequential):
    def forward(self, input, output_size):
        for module in self:
            if isinstance(module, nn.ConvTranspose2d):  # isinstance类似 type() 返回bool
                input = module(input, output_size)
            else:
                input = module(input)
        return input


class CA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.act = nn.ReLU(inplace=False)
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim // 16, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.act(self.fusion(x))
        avg_out = self.globalAvgPool(x)
        avg_out = self.fc(avg_out)
        out = x * avg_out
        return out

class PC3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=dim, out_channels=dim//4, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim//4, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim//2, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x0,x1,x2),dim=1)

        return x


class PC2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=dim, out_channels=dim//4, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim//4, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x = torch.cat((x0,x1),dim=1)

        return x


class EDGE_fusion(nn.Module):

    """
        input:
            EDGE(B*T,G,H,W)---->(B*T,K,H,W),
            FEA(B*T,C0,H,W)----->(B*T,C1,H,W),
        out: B*T,K,H,W
    """

    def __init__(self, group_dim, edge_dim, fea_dim, mid_fea_dim):
        super().__init__()

        self.edge_embed = nn.Sequential(
            nn.Conv2d(in_channels=group_dim, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=8, out_channels=edge_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.fea_down = nn.Sequential(
            nn.Conv2d(in_channels=fea_dim, out_channels=mid_fea_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        self.scale = mid_fea_dim ** -0.5

        self.fusion = nn.Conv2d(in_channels=edge_dim, out_channels=edge_dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, edge, fea):
        edge_ = self.edge_embed(edge)
        fea_ = self.fea_down(fea)

        BT, K, H, W = edge_.shape
        _, C, _, _ = fea_.shape
        edge = edge_.view(BT,K,-1)
        fea = fea_.view(BT,C,-1)

        corr_map = (edge @ fea.transpose(-2, -1)) / self.scale  # BT,K,C
        feav = corr_map @ fea  # BT,K,HW

        feav = feav.view(BT,K,H,W)
        fusion = self.act(self.fusion(feav))

        return fusion


class SynBlock(nn.Module):
    def __init__(self, n_inputs, nf, ks):
        super(SynBlock, self).__init__()
        self.generated_ks = ks

        def KernelNet():
            return MySequential(
                    PC3(nf),
                    torch.nn.ReLU(inplace=False),
                    CA(nf),
                    PC2(nf),
                    torch.nn.ReLU(inplace=False),
                    CA(nf//2),
                    torch.nn.Conv2d(in_channels=nf//2, out_channels=self.generated_ks, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks, out_channels=self.generated_ks, kernel_size=3, stride=1, padding=1)
                )

        def Offsetnet():
            return MySequential(
                    PC3(nf),
                    torch.nn.ReLU(inplace=False),
                    CA(nf),
                    PC2(nf),
                    torch.nn.ReLU(inplace=False),
                    CA(nf//2),
                    torch.nn.Conv2d(in_channels=nf//2, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1)
                )

        def Masknet():
            return MySequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=self.generated_ks ** 2, out_channels=self.generated_ks ** 2, kernel_size=3, stride=1, padding=1),
                    torch.nn.Sigmoid()
                )

        def Biasnet():
            return MySequential(
                    PC3(nf),
                    torch.nn.ReLU(inplace=False),
                    CA(nf),
                    PC2(nf),
                    torch.nn.ReLU(inplace=False),
                    CA(nf//2),
                    torch.nn.Conv2d(in_channels=nf//2, out_channels=3, kernel_size=3, stride=1, padding=1),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
                )

        self.n_inputs = n_inputs
        self.kernel_pad = int(math.floor(self.generated_ks / 2.0))

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])  # 重复填充

        self.ModuleV = KernelNet()
        self.ModuleH = KernelNet()
        self.ModuleOffX = Offsetnet()
        self.ModuleOffY = Offsetnet()
        self.ModuleM = Masknet()
        self.ModuleB = Biasnet()

        self.Edge = EDGE_fusion(group_dim=1, edge_dim=16, fea_dim=nf, mid_fea_dim=nf//2)

        self.feature_fuse = Conv_2d(nf * n_inputs, nf, kernel_size=1, stride=1, batchnorm=False, bias=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, fea, frames, edges, output_size):

        H, W = output_size

        bia = torch.cat(torch.unbind(fea, 1), 1)
        bia = self.lrelu(self.feature_fuse(bia))
        bias = self.ModuleB(bia, (H, W))

        B, C, T, cur_H, cur_W = fea.shape

        _,_,_,eh,ew = edges.shape
        edge_ = edges.transpose(1, 2).view(B*T, 1, eh, ew)
        edges_ = F.interpolate(edge_, size=(cur_H, cur_W), mode='bilinear')

        fea = fea.transpose(1, 2).reshape(B*T, C, cur_H, cur_W)

        edge_fusion = self.Edge(edges_, fea)

        v_s = self.ModuleV(fea, (H, W)).view(B, T, -1, H, W)
        h_s = self.ModuleH(fea, (H, W)).view(B, T, -1, H, W)
        OffX_s = self.ModuleOffX(fea, (H, W)).view(B, T, -1, H, W)
        OffY_s = self.ModuleOffY(fea, (H, W)).view(B, T, -1, H, W)

        M_s = self.ModuleM(edge_fusion, (H, W)).view(B, T, -1, H, W)

        warp = []

        for i in range(self.n_inputs):
            v = v_s[:, i].contiguous()
            h = h_s[:, i].contiguous()
            OffX = OffX_s[:, i].contiguous()
            OffY = OffY_s[:, i].contiguous()
            M = M_s[:, i].contiguous()
            frame = F.interpolate(frames[i], size=v.size()[-2:], mode='bilinear')

            warp.append(
                dsepconv.FunctionDSepconv(self.modulePad(frame), v, h, OffX, OffY, M)
            )
        framet = sum(warp) + bias

        return framet

if __name__ == '__main__':
    model = UNet_3D_3D('unet_18', n_inputs=4, n_outputs=1)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('the number of network parameters: {}'.format(total_params))

