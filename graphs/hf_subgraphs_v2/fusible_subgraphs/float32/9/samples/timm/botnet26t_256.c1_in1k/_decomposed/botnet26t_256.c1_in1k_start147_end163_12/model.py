import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_1 @ in_3
        tmp_1 = tmp_0.reshape(-1, 16, 31)
        tmp_0 = None
        tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
        tmp_1 = None
        tmp_3 = tmp_2.flatten(1)
        tmp_2 = None
        tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None)
        tmp_3 = None
        tmp_5 = tmp_4.reshape(-1, 17, 31)
        tmp_4 = None
        tmp_6 = tmp_5[slice(None, None, None), slice(None, 16, None), slice(15, None, None)]
        tmp_5 = None
        tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
        tmp_6 = None
        tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
        tmp_7 = None
        tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
        tmp_8 = None
        tmp_10 = tmp_9 + in_2
        tmp_9 = None
        tmp_11 = tmp_10.reshape(4, 256, 256)
        tmp_10 = None
        tmp_12 = in_0 + tmp_11
        tmp_11 = None
        tmp_13 = tmp_12.softmax(dim=-1)
        tmp_12 = None
        tmp_14 = tmp_13 @ in_4
        tmp_13 = None
        tmp_15 = tmp_14.transpose(-1, -2)
        tmp_14 = None
        return (tmp_15,)