import torch

class GraphModule(torch.nn.Module):

    def forward(self):
        tmp_0 = torch.eye(13)
        return (tmp_0,)