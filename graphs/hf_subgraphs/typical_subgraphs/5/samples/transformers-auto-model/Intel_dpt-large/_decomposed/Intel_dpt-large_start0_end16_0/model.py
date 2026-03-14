import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = in_5
        tmp_6 = in_6
        tmp_7 = tmp_4[slice(None, None, None), slice(None, 1, None)]
        tmp_8 = tmp_4[0, slice(1, None, None)]
        tmp_4 = None
        tmp_9 = tmp_8.reshape(1, 24, 24, -1)
        tmp_8 = None
        tmp_10 = tmp_9.permute(0, 3, 1, 2)
        tmp_9 = None
        tmp_11 = torch.nn.functional.interpolate(tmp_10, size=(24, 24), mode='bilinear')
        tmp_10 = None
        tmp_12 = tmp_11.permute(0, 2, 3, 1)
        tmp_11 = None
        tmp_13 = tmp_12.reshape(1, 576, -1)
        tmp_12 = None
        tmp_14 = torch.cat([tmp_7, tmp_13], dim=1)
        tmp_7 = tmp_13 = None
        tmp_15 = torch.conv2d(tmp_0, tmp_2, tmp_1, (16, 16), (0, 0), (1, 1), 1)
        tmp_0 = tmp_2 = tmp_1 = None
        tmp_16 = tmp_15.flatten(2)
        tmp_15 = None
        tmp_17 = tmp_16.transpose(1, 2)
        tmp_16 = None
        tmp_18 = tmp_3.expand(1, -1, -1)
        tmp_3 = None
        tmp_19 = torch.cat((tmp_18, tmp_17), dim=1)
        tmp_18 = tmp_17 = None
        tmp_20 = tmp_19 + tmp_14
        tmp_19 = tmp_14 = None
        tmp_21 = torch.nn.functional.dropout(tmp_20, 0.0, False, False)
        tmp_20 = None
        tmp_22 = torch.nn.functional.layer_norm(tmp_21, (1024,), tmp_6, tmp_5, 1e-12)
        tmp_6 = tmp_5 = None
        return (tmp_21, tmp_22)