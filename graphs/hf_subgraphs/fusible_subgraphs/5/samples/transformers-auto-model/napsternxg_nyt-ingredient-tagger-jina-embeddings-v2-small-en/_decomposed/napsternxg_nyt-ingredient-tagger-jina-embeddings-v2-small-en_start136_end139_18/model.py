import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1):
        tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
        tmp_1 = tmp_0 * in_1
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
        tmp_1 = None
        return (tmp_2,)