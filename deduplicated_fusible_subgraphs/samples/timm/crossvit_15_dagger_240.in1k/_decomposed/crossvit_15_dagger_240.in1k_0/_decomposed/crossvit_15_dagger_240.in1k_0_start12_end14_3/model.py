import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.conv2d(in_0, w_1, w_0, (4, 4), (3, 3), (1, 1), 1)
        tmp_1 = torch.nn.functional.relu(tmp_0, inplace=True)
        tmp_0 = None
        return (tmp_1,)