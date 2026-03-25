import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor, in_4 : torch.Tensor):
        conv2d = torch.conv2d(in_0, in_2, in_1, (16, 16), (0, 0), (1, 1), 1);  in_0 = in_2 = in_1 = None
        tmp_6 = conv2d.flatten(2);  conv2d = None
        tmp_7 = tmp_6.transpose(1, 2);  tmp_6 = None
        tmp_8 = in_3.expand(1, -1, -1);  in_3 = None
        tmp_9 = torch.cat((tmp_8, tmp_7), dim = 1);  tmp_8 = tmp_7 = None
        tmp_10 = torch.nn.functional.dropout(tmp_9, 0.0, False, False);  tmp_9 = None
        tmp_11 = in_4[slice(None, 729, None)];  in_4 = None
        tmp_12 = tmp_11.reshape(1, 27, 27, -1);  tmp_11 = None
        tmp_13 = tmp_12.permute(0, 3, 1, 2);  tmp_12 = None
        return (tmp_10, tmp_13)
        