import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = in_0
        tmp_2 = torch.nn.functional.embedding(tmp_1, tmp_0, 2, None, 2.0, False, False)
        tmp_1 = tmp_0 = None
        return (tmp_2,)