import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = tmp_0.index_select(-2, in_0)
        tmp_0 = None
        return (tmp_1,)