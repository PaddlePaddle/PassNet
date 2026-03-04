import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.embedding(in_1, tmp_0, 1, None, 2.0, False, False)
        tmp_0 = None
        tmp_2 = tmp_1 * 1.0
        tmp_1 = None
        return (tmp_2,)