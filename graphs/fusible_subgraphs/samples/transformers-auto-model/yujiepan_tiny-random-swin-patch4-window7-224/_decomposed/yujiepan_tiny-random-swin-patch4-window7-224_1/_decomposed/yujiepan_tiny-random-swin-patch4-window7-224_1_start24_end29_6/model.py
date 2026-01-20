import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, w_0, w_1):
        tmp_0 = in_1.contiguous()
        tmp_1 = tmp_0.view(-1, 28, 28, 16)
        tmp_0 = None
        tmp_2 = tmp_1.view(1, 784, 16)
        tmp_1 = None
        tmp_3 = in_0 + tmp_2
        tmp_2 = None
        tmp_4 = torch.nn.functional.layer_norm(tmp_3, (16,), w_1, w_0, 1e-05)
        return (tmp_3, tmp_4)