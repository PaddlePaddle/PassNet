import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0):
        tmp_0 = torch.cat((in_0, in_0), dim=-1)
        tmp_1 = tmp_0.cos()
        tmp_2 = tmp_1 * 1.0
        tmp_1 = None
        tmp_3 = tmp_0.sin()
        tmp_0 = None
        tmp_4 = tmp_3 * 1.0
        tmp_3 = None
        tmp_5 = tmp_2.to(dtype=torch.bfloat16)
        tmp_2 = None
        tmp_6 = tmp_4.to(dtype=torch.bfloat16)
        tmp_4 = None
        tmp_7 = torch.set_grad_enabled(True)
        tmp_7 = None
        tmp_8 = torch.tensor(45.254833995939045, dtype=torch.bfloat16)
        return (tmp_5, tmp_6, tmp_8)