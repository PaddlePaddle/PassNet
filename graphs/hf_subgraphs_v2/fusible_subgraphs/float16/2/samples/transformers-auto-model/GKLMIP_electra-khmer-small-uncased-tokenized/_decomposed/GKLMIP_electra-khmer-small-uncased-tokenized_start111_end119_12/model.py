import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_0 = in_0 / 8.0;  in_0 = None
        tmp_1 = tmp_0 + in_1;  tmp_0 = in_1 = None
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1);  tmp_1 = None
        tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False);  tmp_2 = None
        matmul = torch.matmul(tmp_3, in_2);  tmp_3 = in_2 = None
        tmp_5 = matmul.permute(0, 2, 1, 3);  matmul = None
        tmp_6 = tmp_5.contiguous();  tmp_5 = None
        tmp_7 = tmp_6.view((16, 128, 512));  tmp_6 = None
        return (tmp_7,)
        