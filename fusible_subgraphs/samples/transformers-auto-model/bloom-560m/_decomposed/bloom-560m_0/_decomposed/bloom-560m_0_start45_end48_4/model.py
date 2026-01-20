import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3):
        tmp_0 = in_0.baddbmm(batch1=in_2, batch2=in_3, beta=1.0, alpha=0.125)
        tmp_1 = tmp_0.view(1, 16, 18, -1)
        tmp_0 = None
        tmp_2 = in_1[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 18, None)]
        return (tmp_1, tmp_2)