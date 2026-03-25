import torch

class GraphModule(torch.nn.Module):

    def forward(self, w_0, in_0):
        tmp_0 = w_0
        tmp_1 = torch.functional.einsum('ibnd,hnd->ibh', in_0, tmp_0)
        tmp_0 = None
        return (tmp_1,)