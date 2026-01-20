import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.cat((in_1, in_0), dim=1)
        tmp_1 = tmp_0[slice(None, None, None), slice(None, 1, None)]
        tmp_2 = tmp_0[slice(None, None, None), slice(1, None, None)]
        tmp_0 = None
        tmp_3 = tmp_2.transpose(1, 2)
        tmp_2 = None
        tmp_4 = tmp_3.view(1, 320, 14, 14)
        tmp_3 = None
        return (tmp_1, tmp_4)