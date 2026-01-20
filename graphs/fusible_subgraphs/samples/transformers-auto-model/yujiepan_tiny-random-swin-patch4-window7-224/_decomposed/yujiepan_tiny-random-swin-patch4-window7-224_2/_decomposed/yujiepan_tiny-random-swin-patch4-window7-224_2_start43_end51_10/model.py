import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1):
        tmp_0 = torch.nn.functional.layer_norm(in_0, (64,), w_1, w_0, 1e-05)
        tmp_1 = tmp_0.view(1, 7, 7, 64)
        tmp_0 = None
        tmp_2 = torch.nn.functional.pad(tmp_1, (0, 0, 0, 0, 0, 0), 'constant', None)
        tmp_1 = None
        tmp_3 = tmp_2.view(1, 1, 7, 1, 7, 64)
        tmp_2 = None
        tmp_4 = tmp_3.permute(0, 1, 3, 2, 4, 5)
        tmp_3 = None
        tmp_5 = tmp_4.contiguous()
        tmp_4 = None
        tmp_6 = tmp_5.view(-1, 7, 7, 64)
        tmp_5 = None
        tmp_7 = tmp_6.view(-1, 49, 64)
        tmp_6 = None
        return (tmp_7,)