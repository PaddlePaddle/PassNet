import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1, in_2):
        matmul = torch.matmul(in_0, in_1);  in_0 = in_1 = None
        tmp_1 = matmul * 1.0;  matmul = None
        tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1, dtype = torch.float32);  tmp_1 = None
        tmp_3 = tmp_2.to(torch.float32);  tmp_2 = None
        tmp_4 = torch.nn.functional.dropout(tmp_3, p = 0.0, training = False);  tmp_3 = None
        to = tmp_4.to(torch.bfloat16);  tmp_4 = None
        matmul_1 = torch.matmul(to, in_2);  to = in_2 = None
        tmp_6 = matmul_1.transpose(1, 2);  matmul_1 = None
        tmp_7 = tmp_6.contiguous();  tmp_6 = None
        tmp_8 = tmp_7.reshape(1, 257, -1);  tmp_7 = None
        tmp_9 = tmp_8.contiguous();  tmp_8 = None
        return (tmp_9,)
        