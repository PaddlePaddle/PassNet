import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2, w_3, in_0):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = w_3
        tmp_4 = torch.nn.functional.gelu(in_0)
        tmp_5 = tmp_4.reshape(1, 124, 2, 512)
        tmp_4 = None
        tmp_6 = tmp_5.reshape(1, 248, 512)
        tmp_5 = None
        tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 0, 1), 'constant', None)
        tmp_6 = None
        tmp_8 = torch.nn.functional.linear(tmp_7, tmp_3, tmp_2)
        tmp_7 = tmp_3 = tmp_2 = None
        tmp_9 = tmp_8.mean(dim=1)
        tmp_8 = None
        tmp_10 = torch.nn.functional.linear(tmp_9, tmp_1, tmp_0)
        tmp_9 = tmp_1 = tmp_0 = None
        return (tmp_10,)