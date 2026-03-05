import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.ops.aten._assert_scalar.default(in_0, "Runtime assertion failed for expression u0 <= 20 on node 'le_1'")
        tmp_0 = None
        return ()