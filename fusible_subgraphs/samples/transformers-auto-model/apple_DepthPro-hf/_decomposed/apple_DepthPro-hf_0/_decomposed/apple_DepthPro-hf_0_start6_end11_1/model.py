import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
        tmp_1 = tmp_0.permute(2, 0, 1)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
        tmp_1 = None
        tmp_3 = torch.cat([tmp_2, in_0, in_1], dim=0)
        tmp_2 = None
        tmp_4 = tmp_3.to(dtype=torch.float16)
        tmp_3 = None
        return (tmp_4,)