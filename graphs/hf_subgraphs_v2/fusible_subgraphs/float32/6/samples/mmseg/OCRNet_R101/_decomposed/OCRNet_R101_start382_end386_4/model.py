import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
        tmp_1 = in_0.reshape(16, 256, -1)
        tmp_2 = tmp_0.reshape(16, 256, -1)
        tmp_0 = None
        tmp_3 = tmp_2.permute(0, 2, 1)
        tmp_2 = None
        return (tmp_3, tmp_1)