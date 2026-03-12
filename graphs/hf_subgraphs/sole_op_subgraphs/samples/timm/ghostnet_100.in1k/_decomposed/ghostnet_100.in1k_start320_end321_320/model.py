import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.dropout(in_0, p=0.2, training=False)
        return (tmp_0,)