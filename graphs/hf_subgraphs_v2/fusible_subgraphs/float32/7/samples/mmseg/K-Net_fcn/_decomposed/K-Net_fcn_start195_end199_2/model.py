import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.functional.einsum('bnhw,bchw->bnc', in_1, in_2)
        tmp_1 = in_0.reshape(1, 150, 512, -1)
        tmp_2 = tmp_1.permute(0, 1, 3, 2)
        tmp_1 = None
        tmp_3 = tmp_0.reshape(-1, 256)
        tmp_0 = None
        return (tmp_2, tmp_3)