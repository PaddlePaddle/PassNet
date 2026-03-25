import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6):
        conv2d = torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), 57);  in_5 = in_1 = in_0 = None
        tmp_3 = torch.cat([in_2, in_3, conv2d], dim = 1);  in_2 = in_3 = conv2d = None
        tmp_4 = tmp_3.reshape(1, 8, 19, 196);  tmp_3 = None
        tmp_5 = tmp_4.transpose(-1, -2);  tmp_4 = None
        tmp_6 = in_6 * tmp_5;  in_6 = tmp_5 = None
        tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None);  tmp_6 = None
        tmp_8 = 0.22941573387056177 * in_4;  in_4 = None
        tmp_9 = tmp_8 + tmp_7;  tmp_8 = tmp_7 = None
        tmp_10 = tmp_9.transpose(1, 2);  tmp_9 = None
        tmp_11 = tmp_10.reshape(1, 197, 152);  tmp_10 = None
        return (tmp_11,)
        