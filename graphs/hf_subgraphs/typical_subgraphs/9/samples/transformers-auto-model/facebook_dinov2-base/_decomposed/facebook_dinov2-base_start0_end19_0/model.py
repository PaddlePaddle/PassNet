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
        tmp_7 = tmp_0.to(dtype=torch.float32)
        tmp_0 = None
        tmp_8 = torch.conv2d(tmp_7, tmp_2, tmp_1, (14, 14), (0, 0), (1, 1), 1)
        tmp_7 = tmp_2 = tmp_1 = None
        tmp_9 = tmp_8.flatten(2)
        tmp_8 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_12 = torch.cat((tmp_11, tmp_10), dim=1)
        tmp_11 = tmp_10 = None
        tmp_13 = tmp_4[slice(None, None, None), slice(None, 1, None)]
        tmp_14 = tmp_4[slice(None, None, None), slice(1, None, None)]
        tmp_4 = None
        tmp_15 = tmp_14.reshape(1, 37, 37, 768)
        tmp_14 = None
        tmp_16 = tmp_15.permute(0, 3, 1, 2)
        tmp_15 = None
        tmp_17 = tmp_16.to(torch.float32)
        tmp_16 = None
        tmp_18 = torch.nn.functional.interpolate(tmp_17, size=(16, 16), mode='bicubic', align_corners=False)
        tmp_17 = None
        tmp_19 = tmp_18.to(dtype=torch.float32)
        tmp_18 = None
        tmp_20 = tmp_19.permute(0, 2, 3, 1)
        tmp_19 = None
        tmp_21 = tmp_20.view(1, -1, 768)
        tmp_20 = None
        tmp_22 = torch.cat((tmp_13, tmp_21), dim=1)
        tmp_13 = tmp_21 = None
        tmp_23 = tmp_12 + tmp_22
        tmp_12 = tmp_22 = None
        tmp_24 = torch.nn.functional.dropout(tmp_23, 0.0, False, False)
        tmp_23 = None
        tmp_25 = torch.nn.functional.layer_norm(tmp_24, (768,), tmp_6, tmp_5, 1e-06)
        tmp_6 = tmp_5 = None
        return (tmp_24, tmp_25)