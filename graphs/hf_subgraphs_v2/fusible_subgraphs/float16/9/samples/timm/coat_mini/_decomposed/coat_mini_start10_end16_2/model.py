import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5):
        conv2d = torch.conv2d(in_5, in_1, in_0, (1, 1), (1, 1), (1, 1), 152);  in_1 = in_0 = None
        tmp_5 = conv2d + in_5;  conv2d = in_5 = None
        tmp_6 = tmp_5.flatten(2);  tmp_5 = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        tmp_8 = torch.cat((in_4, tmp_7), dim = 1);  in_4 = tmp_7 = None
        tmp_9 = torch.nn.functional.layer_norm(tmp_8, (152,), in_3, in_2, 1e-06);  in_3 = in_2 = None
        return (tmp_8, tmp_9)
        