import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (1, 1), (1, 1), 768)
        tmp_0 = None
        tmp_2 = tmp_1.mean((2, 3), keepdim=True)
        return (tmp_1, tmp_2)