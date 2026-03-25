import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        tmp_0 = torch.nn.functional.relu(in_2, inplace = True);  in_2 = None
        tmp_1 = in_1.reshape(16, 256, -1);  in_1 = None
        tmp_2 = tmp_0.reshape(16, 256, -1);  tmp_0 = None
        tmp_3 = tmp_2.permute(0, 2, 1);  tmp_2 = None
        tmp_4 = tmp_3.contiguous();  tmp_3 = None
        matmul = torch.matmul(in_0, tmp_1);  in_0 = tmp_1 = None
        tmp_6 = 0.0625 * matmul;  matmul = None
        tmp_7 = torch.nn.functional.softmax(tmp_6, dim = -1);  tmp_6 = None
        matmul_1 = torch.matmul(tmp_7, tmp_4);  tmp_7 = tmp_4 = None
        tmp_9 = matmul_1.permute(0, 2, 1);  matmul_1 = None
        tmp_10 = tmp_9.contiguous();  tmp_9 = None
        tmp_11 = tmp_10.reshape(16, -1, 64, 128);  tmp_10 = None
        return (tmp_11,)
        