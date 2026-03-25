import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.max(in_0, -1, keepdim=True)
        tmp_1 = tmp_0[0]
        tmp_0 = None
        tmp_2 = tmp_1.expand_as(in_0)
        tmp_1 = None
        tmp_3 = tmp_2 - in_0
        tmp_2 = None
        tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
        tmp_3 = None
        tmp_5 = in_1.view(4, 512, -1)
        return (tmp_4, tmp_5)