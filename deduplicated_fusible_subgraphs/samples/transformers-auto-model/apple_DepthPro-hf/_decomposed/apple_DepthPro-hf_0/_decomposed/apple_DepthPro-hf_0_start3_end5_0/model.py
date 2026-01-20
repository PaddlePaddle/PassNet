import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.unfold(in_0, kernel_size=(384, 384), stride=(192, 192))
        tmp_1 = tmp_0.permute(2, 0, 1)
        tmp_0 = None
        return (tmp_1,)