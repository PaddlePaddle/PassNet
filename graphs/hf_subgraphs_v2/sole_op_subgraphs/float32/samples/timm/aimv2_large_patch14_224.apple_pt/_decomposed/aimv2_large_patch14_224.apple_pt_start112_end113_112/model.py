import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.rms_norm(in_0, (1024,), tmp_0, 1e-05)
        tmp_0 = None
        return (tmp_1,)