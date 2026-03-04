import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
        tmp_1 = tmp_0.reshape(1, -1)
        tmp_0 = None
        tmp_2 = tmp_1.view(1, -1, 1, 1)
        tmp_1 = None
        tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
        tmp_2 = None
        tmp_4 = tmp_3 * in_0
        tmp_3 = None
        tmp_5 = torch.sum(tmp_4, dim=1)
        tmp_4 = None
        tmp_6 = tmp_5.contiguous()
        tmp_5 = None
        return (tmp_6,)