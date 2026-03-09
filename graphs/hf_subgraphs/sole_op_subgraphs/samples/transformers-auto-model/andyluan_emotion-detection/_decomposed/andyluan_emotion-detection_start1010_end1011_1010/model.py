import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = in_0.baddbmm(batch1=in_2, batch2=in_1, beta=1.0, alpha=0.125)
        return (tmp_0,)