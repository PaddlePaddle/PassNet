import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = torch.nn.functional.embedding(tmp_0, tmp_1, None, None, 2.0, False, False)
        tmp_0 = tmp_1 = None
        return (tmp_2,)