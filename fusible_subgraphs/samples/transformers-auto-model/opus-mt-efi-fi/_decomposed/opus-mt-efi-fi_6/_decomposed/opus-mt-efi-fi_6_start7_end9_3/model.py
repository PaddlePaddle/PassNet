import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.nn.functional.embedding(in_0, w_0, 61192, None, 2.0, False, False)
        tmp_1 = tmp_0 * 22.627416997969522
        tmp_0 = None
        return (tmp_1,)