import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.batch_norm(in_1, None, None, training=True, momentum=0.0, eps=in_0)
        return (tmp_0,)