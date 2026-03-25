import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = torch.nn.functional.embedding(tmp_0, tmp_1, 9000, None, 2.0, False, False)
        tmp_0 = tmp_1 = None
        tmp_3 = tmp_2 * 22.627416997969522
        tmp_2 = None
        return (tmp_3,)