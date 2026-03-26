import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3, in_4):
        conv2d = torch.conv2d(in_4, in_1, in_0, (1, 1), (1, 1), (1, 1), 1);  in_4 = in_1 = in_0 = None
        tmp_3 = torch.sigmoid(conv2d);  conv2d = None
        tmp_4 = tmp_3[(slice(None, None, None), 0, slice(None, None, None), slice(None, None, None))]
        tmp_5 = tmp_4.unsqueeze(1);  tmp_4 = None
        tmp_6 = in_3 * tmp_5;  in_3 = tmp_5 = None
        tmp_7 = tmp_3[(slice(None, None, None), 1, slice(None, None, None), slice(None, None, None))];  tmp_3 = None
        tmp_8 = tmp_7.unsqueeze(1);  tmp_7 = None
        tmp_9 = in_2 * tmp_8;  in_2 = tmp_8 = None
        tmp_10 = tmp_6 + tmp_9;  tmp_6 = tmp_9 = None
        return (tmp_10,)
        