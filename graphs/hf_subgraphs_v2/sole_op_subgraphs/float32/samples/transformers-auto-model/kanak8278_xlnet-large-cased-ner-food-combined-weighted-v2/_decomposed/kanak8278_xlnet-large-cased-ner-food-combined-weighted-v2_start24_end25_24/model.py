import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.one_hot(in_0, num_classes=2)
        return (tmp_0,)