import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        conv2d = torch.conv2d(in_7, in_5, None, (1, 1), (2, 2), (2, 2), 32);  in_7 = in_5 = None
        tmp_7 = torch.cat([in_6, conv2d], 1);  in_6 = conv2d = None
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001);  tmp_7 = in_1 = in_2 = in_4 = in_3 = None
        tmp_9 = torch.prelu(tmp_8, in_0);  tmp_8 = in_0 = None
        return (tmp_9,)
        