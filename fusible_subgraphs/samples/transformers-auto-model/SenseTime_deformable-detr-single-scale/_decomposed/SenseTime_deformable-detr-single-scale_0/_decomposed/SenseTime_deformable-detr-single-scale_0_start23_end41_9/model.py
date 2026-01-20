import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_1 / in_4
        tmp_1 = in_3 + tmp_0
        tmp_0 = None
        tmp_2 = in_0.split([625], dim=1)
        tmp_3 = tmp_2[0]
        tmp_2 = None
        tmp_4 = 2 * tmp_1
        tmp_1 = None
        tmp_5 = tmp_4 - 1
        tmp_4 = None
        tmp_6 = tmp_3.flatten(2)
        tmp_3 = None
        tmp_7 = tmp_6.transpose(1, 2)
        tmp_6 = None
        tmp_8 = tmp_7.reshape(8, 32, 25, 25)
        tmp_7 = None
        tmp_9 = tmp_5[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0]
        tmp_5 = None
        tmp_10 = tmp_9.transpose(1, 2)
        tmp_9 = None
        tmp_11 = tmp_10.flatten(0, 1)
        tmp_10 = None
        tmp_12 = torch.nn.functional.grid_sample(tmp_8, tmp_11, mode='bilinear', padding_mode='zeros', align_corners=False)
        tmp_8 = tmp_11 = None
        tmp_13 = in_2.transpose(1, 2)
        tmp_14 = tmp_13.reshape(8, 1, 625, 4)
        tmp_13 = None
        tmp_15 = torch.stack([tmp_12], dim=-2)
        tmp_12 = None
        tmp_16 = tmp_15.flatten(-2)
        tmp_15 = None
        tmp_17 = tmp_16 * tmp_14
        tmp_16 = tmp_14 = None
        return (tmp_17,)