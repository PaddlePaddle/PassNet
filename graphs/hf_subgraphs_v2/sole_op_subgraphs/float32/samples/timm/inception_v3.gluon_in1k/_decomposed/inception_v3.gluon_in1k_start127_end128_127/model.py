import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.avg_pool2d(in_0, kernel_size=3, stride=1, padding=1)
        return (tmp_0,)