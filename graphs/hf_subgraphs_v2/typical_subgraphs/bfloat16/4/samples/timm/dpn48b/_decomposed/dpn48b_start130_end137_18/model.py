import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8):
        tmp_6 = torch.nn.functional.relu(in_8, inplace = True);  in_8 = None
        to = tmp_6.to(torch.bfloat16)
        conv2d = torch.conv2d(to, in_0, None, (1, 1), (0, 0), (1, 1), 1);  to = in_0 = None
        to_1 = tmp_6.to(torch.bfloat16);  tmp_6 = None
        conv2d_1 = torch.conv2d(to_1, in_1, None, (1, 1), (0, 0), (1, 1), 1);  to_1 = in_1 = None
        tmp_9 = in_7 + conv2d;  in_7 = conv2d = None
        tmp_10 = torch.cat([in_6, conv2d_1], dim = 1);  in_6 = conv2d_1 = None
        tmp_11 = torch.cat((tmp_9, tmp_10), dim = 1)
        tmp_12 = torch.nn.functional.batch_norm(tmp_11, in_2, in_3, in_5, in_4, False, 0.1, 0.001);  tmp_11 = in_2 = in_3 = in_5 = in_4 = None
        return (tmp_10, tmp_9, tmp_12)
        