import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        linear = torch.nn.functional.linear(in_2, in_1, in_0);  in_2 = in_1 = in_0 = None
        tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False);  linear = None
        tmp_4 = tmp_3.view(-1, 12, 12, 1536);  tmp_3 = None
        tmp_5 = tmp_4.view(-1, 1, 1, 12, 12, 1536);  tmp_4 = None
        tmp_6 = tmp_5.permute(0, 1, 3, 2, 4, 5);  tmp_5 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        tmp_8 = tmp_7.view(-1, 12, 12, 1536);  tmp_7 = None
        tmp_9 = tmp_8.view(1, 144, 1536);  tmp_8 = None
        return (tmp_9,)
        