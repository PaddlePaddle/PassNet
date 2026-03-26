import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.meshgrid(in_0, in_1, indexing='ij')
        return (tmp_0,)