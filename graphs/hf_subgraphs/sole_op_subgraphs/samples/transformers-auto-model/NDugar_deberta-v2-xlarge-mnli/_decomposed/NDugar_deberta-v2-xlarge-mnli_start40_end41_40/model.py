import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, w_1, w_2):
        tmp_0 = w_0
        tmp_1 = w_1
        tmp_2 = w_2
        tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1536,), tmp_1, tmp_0, 1e-07)
        tmp_2 = tmp_1 = tmp_0 = None
        return (tmp_3,)