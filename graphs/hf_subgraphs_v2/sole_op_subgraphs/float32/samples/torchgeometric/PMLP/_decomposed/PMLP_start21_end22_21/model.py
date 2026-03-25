import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.batch_norm(in_0, None, None, None, None, True, 0.1, 1e-05)
        return (tmp_0,)