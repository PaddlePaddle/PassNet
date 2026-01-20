import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (768,), w_1, w_0, 1e-05)
        tmp_1 = torch.nn.functional.dropout(tmp_0, 0.1, False, False)
        tmp_0 = None
        tmp_2 = torch.rand([])
        tmp_2 = None
        return (tmp_1,)