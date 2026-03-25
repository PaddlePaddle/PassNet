import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = torch.nn.functional.embedding(tmp_0, tmp_1, 61482, None, 2.0, False, False)
        tmp_0 = tmp_1 = None
        return (tmp_2,)