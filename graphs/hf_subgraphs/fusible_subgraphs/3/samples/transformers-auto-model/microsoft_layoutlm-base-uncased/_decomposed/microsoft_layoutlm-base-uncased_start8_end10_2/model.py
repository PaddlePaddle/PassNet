import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0
        tmp_1 = torch.nn.functional.embedding(in_1, tmp_0, None, None, 2.0, False, False)
        tmp_0 = None
        tmp_2 = in_2[slice(None, None, None), slice(None, None, None), 0]
        return (tmp_2, tmp_1)