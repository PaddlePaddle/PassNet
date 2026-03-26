import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
        tmp_1 = tmp_0.view(24, 512, -1)
        tmp_0 = None
        tmp_2 = tmp_1.permute(0, 2, 1)
        tmp_1 = None
        tmp_3 = tmp_2.contiguous()
        tmp_2 = None
        return (tmp_3,)