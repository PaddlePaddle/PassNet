import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0):
        tmp_0 = torch.nn.functional.embedding(in_0, w_0, 1, None, 2.0, False, False)
        tmp_1 = tmp_0 * 45.254833995939045
        tmp_0 = None
        return (tmp_1,)