import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_1 = torch.nn.functional.pad(in_2, [0, 1], 'constant', None);  in_2 = None
        tmp_2 = tmp_1.flatten(1);  tmp_1 = None
        tmp_3 = torch.nn.functional.pad(tmp_2, [0, 7], 'constant', None);  tmp_2 = None
        tmp_4 = tmp_3.reshape(-1, 9, 15);  tmp_3 = None
        tmp_5 = tmp_4[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))];  tmp_4 = None
        tmp_6 = tmp_5.reshape(4, 8, 1, 8, 8);  tmp_5 = None
        tmp_7 = tmp_6.expand(-1, -1, 8, -1, -1);  tmp_6 = None
        tmp_8 = tmp_7.permute((0, 1, 3, 2, 4));  tmp_7 = None
        tmp_9 = in_1.transpose(1, 2);  in_1 = None
        tmp_10 = in_0.transpose(-1, -2);  in_0 = None
        return (tmp_9, tmp_8, tmp_10)
        