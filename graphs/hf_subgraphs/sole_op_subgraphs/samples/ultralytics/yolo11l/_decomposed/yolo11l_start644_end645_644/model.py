import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.ops.aten._local_scalar_dense(in_0)
        return (tmp_0,)