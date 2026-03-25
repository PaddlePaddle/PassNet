import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9):
        conv2d = torch.conv2d(in_7, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_1 = in_0 = None
        tmp_8 = conv2d + in_9;  conv2d = in_9 = None
        tmp_9 = in_2.unsqueeze(-1);  in_2 = None
        tmp_10 = tmp_9.unsqueeze(-1);  tmp_9 = None
        tmp_11 = tmp_10 * tmp_8;  tmp_10 = tmp_8 = None
        tmp_12 = in_8 + tmp_11;  in_8 = tmp_11 = None
        tmp_13 = torch.nn.functional.batch_norm(tmp_12, in_3, in_4, in_6, in_5, False, 0.1, 1e-05);  in_3 = in_4 = in_6 = in_5 = None
        return (tmp_13, tmp_12)
        