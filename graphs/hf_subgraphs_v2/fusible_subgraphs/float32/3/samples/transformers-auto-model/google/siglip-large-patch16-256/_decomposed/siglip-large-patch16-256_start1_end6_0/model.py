import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = torch.conv2d(in_4, tmp_2, tmp_1, (16, 16), 'valid', (1, 1), 1)
        tmp_2 = tmp_1 = None
        tmp_5 = tmp_4.flatten(2)
        tmp_4 = None
        tmp_6 = tmp_5.transpose(1, 2)
        tmp_5 = None
        tmp_7 = torch.nn.functional.embedding(tmp_0, tmp_3, None, None, 2.0, False, False)
        tmp_0 = tmp_3 = None
        tmp_8 = tmp_6 + tmp_7
        tmp_6 = tmp_7 = None
        return (tmp_8,)