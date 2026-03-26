import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        conv2d = torch.conv2d(in_7, in_5, in_4, (1, 1), (0, 0), (1, 1), 128);  in_5 = in_4 = None
        tmp_7 = in_6 + conv2d;  in_6 = conv2d = None
        tmp_8 = tmp_7 + in_7;  tmp_7 = in_7 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_0, in_1, in_3, in_2, False, 0.1, 1e-05);  tmp_8 = in_0 = in_1 = in_3 = in_2 = None
        tmp_10 = tmp_9.mean((2, 3), keepdim = True)
        return (tmp_9, tmp_10)
        