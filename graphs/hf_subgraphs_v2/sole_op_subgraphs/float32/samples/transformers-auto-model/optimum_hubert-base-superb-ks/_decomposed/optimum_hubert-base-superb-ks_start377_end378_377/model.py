import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12):
        tmp_0 = torch.stack((in_5, in_6, in_7, in_8, in_9, in_10, in_11, in_12, in_0, in_1, in_2, in_3, in_4), dim=1)
        return (tmp_0,)