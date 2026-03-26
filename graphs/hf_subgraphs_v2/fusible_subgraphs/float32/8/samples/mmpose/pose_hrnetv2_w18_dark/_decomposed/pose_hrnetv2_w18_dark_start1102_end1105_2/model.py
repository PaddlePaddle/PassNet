import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        in_2 += in_0
        tmp_0 = in_2
        tmp_0 += in_1
        tmp_1 = tmp_0
        tmp_0 = None
        tmp_2 = torch.nn.functional.relu(tmp_1, inplace=True)
        tmp_1 = None
        return (tmp_2,)