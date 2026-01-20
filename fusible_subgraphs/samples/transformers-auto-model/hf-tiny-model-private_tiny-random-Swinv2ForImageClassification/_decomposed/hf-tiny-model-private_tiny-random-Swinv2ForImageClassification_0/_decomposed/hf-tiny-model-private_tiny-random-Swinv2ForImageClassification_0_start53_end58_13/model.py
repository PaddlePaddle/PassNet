import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (32,), w_1, w_0, 1e-05)
        tmp_1 = tmp_0.view(1, 8, 8, 32)
        tmp_2 = torch.nn.functional.pad(tmp_1, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_1 = None
        tmp_3 = tmp_2.view(1, 4, 2, 4, 2, 32)
        tmp_2 = None
        tmp_4 = tmp_3.permute(0, 1, 3, 2, 4, 5)
        tmp_3 = None
        return (tmp_0, tmp_4)