import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1):
        tmp_0 = torch.nn.functional.relu(in_1, inplace = True);  in_1 = None
        tmp_1 = in_0.view(12, 21, -1);  in_0 = None
        tmp_2 = tmp_0.view(12, 512, -1)
        tmp_3 = tmp_2.permute(0, 2, 1);  tmp_2 = None
        tmp_4 = 1 * tmp_1;  tmp_1 = None
        tmp_5 = torch.nn.functional.softmax(tmp_4, dim = 2);  tmp_4 = None
        matmul = torch.matmul(tmp_5, tmp_3);  tmp_5 = tmp_3 = None
        tmp_7 = matmul.permute(0, 2, 1);  matmul = None
        tmp_8 = tmp_7.contiguous();  tmp_7 = None
        tmp_9 = tmp_8.unsqueeze(3);  tmp_8 = None
        return (tmp_9, tmp_0)
        