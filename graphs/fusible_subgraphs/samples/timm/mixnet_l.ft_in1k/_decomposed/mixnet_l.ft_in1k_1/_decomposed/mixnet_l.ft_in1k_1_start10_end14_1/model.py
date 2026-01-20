import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, w_0, w_1, w_2, w_3):
        tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
        tmp_1 = torch.nn.functional.batch_norm(tmp_0, w_0, w_1, w_3, w_2, False, 0.1, 1e-05)
        tmp_0 = None
        tmp_2 = torch.nn.functional.silu(tmp_1, inplace=True)
        tmp_1 = None
        tmp_3 = tmp_2.mean((2, 3), keepdim=True)
        return (tmp_2, tmp_3)