import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, w_2, w_3, w_4, w_5):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = w_2
        tmp_4 = w_3
        tmp_5 = w_4
        tmp_6 = w_5
        tmp_7 = torch.nn.functional.interpolate(tmp_0, scale_factor=0.25, mode='bilinear', align_corners=False)
        tmp_8 = torch.nn.functional.interpolate(tmp_0, scale_factor=0.5, mode='bilinear', align_corners=False)
        tmp_9 = torch.nn.functional.interpolate(tmp_0, scale_factor=1, mode='bilinear', align_corners=False)
        tmp_0 = None
        tmp_10 = torch.nn.functional.unfold(tmp_8, kernel_size=(384, 384), stride=(192, 192))
        tmp_8 = None
        tmp_11 = tmp_10.permute(2, 0, 1)
        tmp_10 = None
        tmp_12 = tmp_11.reshape(-1, 3, 384, 384)
        tmp_11 = None
        tmp_13 = torch.nn.functional.unfold(tmp_9, kernel_size=(384, 384), stride=(288, 288))
        tmp_9 = None
        tmp_14 = tmp_13.permute(2, 0, 1)
        tmp_13 = None
        tmp_15 = tmp_14.reshape(-1, 3, 384, 384)
        tmp_14 = None
        tmp_16 = torch.cat([tmp_15, tmp_12, tmp_7], dim=0)
        tmp_15 = tmp_12 = tmp_7 = None
        tmp_17 = tmp_16.to(dtype=torch.float16)
        tmp_16 = None
        tmp_18 = torch.conv2d(tmp_17, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_17 = tmp_2 = tmp_1 = None
        tmp_19 = tmp_18.flatten(2)
        tmp_18 = None
        tmp_20 = tmp_19.transpose(1, 2)
        tmp_19 = None
        tmp_21 = tmp_3.expand(35, -1, -1)
        tmp_3 = None
        tmp_22 = torch.cat((tmp_21, tmp_20), dim=1)
        tmp_21 = tmp_20 = None
        tmp_23 = tmp_22 + tmp_4
        tmp_22 = tmp_4 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.0, False, False)
        tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (1024,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        return (tmp_24, tmp_25)