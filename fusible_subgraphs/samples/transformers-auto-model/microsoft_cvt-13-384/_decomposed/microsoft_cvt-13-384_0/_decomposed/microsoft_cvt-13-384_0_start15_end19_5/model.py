import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.functional.einsum('bhlt,bhtv->bhlv', [in_1, in_0])
        tmp_1 = tmp_0.permute(0, 2, 1, 3)
        tmp_0 = None
        tmp_2 = tmp_1.contiguous()
        tmp_1 = None
        tmp_3 = tmp_2.view(1, 9216, 64)
        tmp_2 = None
        return (tmp_3,)