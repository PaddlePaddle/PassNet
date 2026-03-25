import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2, in_3):
        tmp_2 = torch.nn.functional.dropout(in_3, 0, False, False);  in_3 = None
        tmp_3 = tmp_2 + in_2;  tmp_2 = in_2 = None
        tmp_4 = torch.nn.functional.relu(tmp_3, inplace = False);  tmp_3 = None
        conv2d = torch.conv2d(tmp_4, in_1, in_0, (1, 1), (1, 0), (1, 1), 1);  in_1 = in_0 = None
        tmp_6 = torch.nn.functional.relu(conv2d, inplace = False);  conv2d = None
        return (tmp_4, tmp_6)
        