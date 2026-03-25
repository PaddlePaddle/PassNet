import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.nn.functional.softmax(in_0, dim=-1, dtype=torch.float32)
        tmp_1 = tmp_0.to(torch.float32)
        tmp_0 = None
        tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
        tmp_1 = None
        return (tmp_2,)