import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = torch.nn.functional.relu(in_3, inplace=True)
        tmp_1 = tmp_0 + in_2
        tmp_2 = tmp_1 + in_1
        tmp_3 = tmp_2 + in_0
        return (tmp_1, tmp_2, tmp_3, tmp_0)