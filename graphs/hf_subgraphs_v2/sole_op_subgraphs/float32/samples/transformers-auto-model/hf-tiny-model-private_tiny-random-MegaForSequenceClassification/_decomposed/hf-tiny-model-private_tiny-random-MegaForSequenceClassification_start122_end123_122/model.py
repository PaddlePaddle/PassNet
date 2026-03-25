import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0):
        tmp_0 = torch.nn.functional.dropout(w_0, p=0.1, training=False)
        return (tmp_0,)