import torch

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2):
        tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
        tmp_1 = tmp_0.permute(2, 0, 1)
        tmp_0 = None
        tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
        tmp_1 = None
        tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
        tmp_4 = tmp_3.permute(2, 0, 1)
        tmp_3 = None
        tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
        tmp_4 = None
        tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
        tmp_5 = tmp_2 = None
        tmp_7 = tmp_6.to(dtype=torch.float16)
        tmp_6 = None
        return (tmp_7,)