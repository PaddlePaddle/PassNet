import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.unfold(in_0, kernel_size=(384, 384), stride=(288, 288))
        return (tmp_0,)