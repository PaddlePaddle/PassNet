import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.embedding(tmp_0, tmp_1, 0, None, 2.0, False, False)
        tmp_0 = tmp_1 = None
        tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None)]
        return (tmp_3, tmp_2)