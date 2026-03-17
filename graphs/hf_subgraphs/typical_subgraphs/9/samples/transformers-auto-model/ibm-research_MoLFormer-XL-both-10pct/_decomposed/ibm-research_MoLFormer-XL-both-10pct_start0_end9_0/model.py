import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, w_0, w_1, in_1):
        tmp_0 = in_0
        tmp_1 = w_0
        tmp_2 = w_1
        tmp_3 = in_1
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), tmp_2, tmp_1, 1e-12)
        tmp_3 = tmp_2 = tmp_1 = None
        tmp_5 = tmp_0.unsqueeze(-1)
        tmp_0 = None
        tmp_6 = tmp_5.expand_as(tmp_4)
        tmp_5 = None
        tmp_7 = tmp_6.float()
        tmp_6 = None
        tmp_8 = tmp_4 * tmp_7
        tmp_9 = torch.sum(tmp_8, dim=1)
        tmp_8 = None
        tmp_10 = tmp_7.sum(dim=1)
        tmp_7 = None
        tmp_11 = torch.clamp(tmp_10, min=1e-09)
        tmp_10 = None
        tmp_12 = tmp_9 / tmp_11
        tmp_9 = tmp_11 = None
        return (tmp_4, tmp_12)