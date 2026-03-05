import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.fold(in_0, output_size=(32, 32), kernel_size=(2, 2), stride=(2, 2))
        return (tmp_0,)