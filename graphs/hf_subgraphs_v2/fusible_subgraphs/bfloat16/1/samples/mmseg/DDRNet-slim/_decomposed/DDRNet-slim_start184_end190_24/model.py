import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
        conv2d = torch.conv2d(in_6, in_0, None, (1, 1), (0, 0), (1, 1), 1);  in_6 = in_0 = None
        tmp_6 = in_5 + conv2d;  in_5 = conv2d = None
        tmp_7 = torch.nn.functional.interpolate(tmp_6, (64, 64), None, 'bilinear', False);  tmp_6 = None
        tmp_8 = in_7 + tmp_7;  in_7 = tmp_7 = None
        tmp_9 = torch.nn.functional.batch_norm(tmp_8, in_1, in_2, in_4, in_3, False, 0.1, 1e-05);  tmp_8 = in_1 = in_2 = in_4 = in_3 = None
        tmp_10 = torch.nn.functional.relu(tmp_9, inplace = True);  tmp_9 = None
        return (tmp_10,)
        