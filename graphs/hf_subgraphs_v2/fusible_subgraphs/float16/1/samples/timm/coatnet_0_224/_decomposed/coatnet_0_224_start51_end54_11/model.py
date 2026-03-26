import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        conv2d = torch.conv2d(in_7, in_1, in_0, (1, 1), (0, 0), (1, 1), 1);  in_7 = in_1 = in_0 = None
        tmp_7 = conv2d + in_6;  conv2d = in_6 = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_2, in_3, in_5, in_4, False, 0.1, 1e-05);  in_2 = in_3 = in_5 = in_4 = None
        return (tmp_7, tmp_8)
        