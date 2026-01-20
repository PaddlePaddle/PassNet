import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0):
        tmp_0 = torch.nn.functional.pad(in_1, [0, 1], 'constant', None)
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
        tmp_7 = tmp_6.permute((0, 1, 3, 2, 4))
        tmp_6 = None
        tmp_8 = in_0.transpose(1, 2)
        tmp_9 = w_0.transpose(-1, -2)
        return (tmp_7, tmp_8, tmp_9)