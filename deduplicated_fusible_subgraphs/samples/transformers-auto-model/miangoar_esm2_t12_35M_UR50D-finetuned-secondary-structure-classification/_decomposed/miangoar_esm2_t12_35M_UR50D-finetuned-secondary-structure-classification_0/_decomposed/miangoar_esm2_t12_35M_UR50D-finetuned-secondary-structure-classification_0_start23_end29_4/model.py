import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, w_0, w_1):
        tmp_0 = in_1 / in_2
        tmp_1 = tmp_0.to(torch.float32)
        tmp_0 = None
        tmp_2 = in_0.unsqueeze(-1)
        tmp_3 = tmp_1 * tmp_2
        tmp_1 = tmp_2 = None
        tmp_4 = tmp_3.to(torch.float32)
        tmp_3 = None
        tmp_5 = torch.nn.functional.layer_norm(tmp_4, (480,), w_1, w_0, 1e-05)
        return (tmp_4, tmp_5)