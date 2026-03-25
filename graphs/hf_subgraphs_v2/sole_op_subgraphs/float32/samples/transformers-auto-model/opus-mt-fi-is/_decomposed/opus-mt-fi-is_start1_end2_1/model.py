import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.nn.functional.embedding(in_0, tmp_0, 60258, None, 2.0, False, False)
        tmp_0 = None
        return (tmp_1,)