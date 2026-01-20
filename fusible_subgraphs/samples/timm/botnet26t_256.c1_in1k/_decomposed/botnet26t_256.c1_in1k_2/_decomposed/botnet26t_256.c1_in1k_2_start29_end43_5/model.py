import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.pad(in_3, [0, 1], 'constant', None)
        tmp_1 = tmp_0.flatten(1)
        tmp_0 = None
        tmp_2 = torch.nn.functional.pad(tmp_1, [0, 7], 'constant', None)
        tmp_1 = None
        tmp_3 = tmp_2.reshape(-1, 9, 15)
        tmp_2 = None
        tmp_4 = tmp_3[slice(None, None, None), slice(None, 8, None), slice(7, None, None)]
        tmp_3 = None
        tmp_5 = tmp_4.reshape(4, 8, 1, 8, 8)
        tmp_4 = None
        tmp_6 = tmp_5.expand(-1, -1, 8, -1, -1)
        tmp_5 = None
        tmp_7 = tmp_6.permute((0, 3, 1, 4, 2))
        tmp_6 = None
        tmp_8 = tmp_7 + in_2
        tmp_7 = None
        tmp_9 = tmp_8.reshape(4, 64, 64)
        tmp_8 = None
        tmp_10 = in_1 + tmp_9
        tmp_9 = None
        tmp_11 = tmp_10.softmax(dim=-1)
        tmp_10 = None
        tmp_12 = tmp_11 @ in_0
        tmp_11 = None
        tmp_13 = tmp_12.transpose(-1, -2)
        tmp_12 = None
        return (tmp_13,)