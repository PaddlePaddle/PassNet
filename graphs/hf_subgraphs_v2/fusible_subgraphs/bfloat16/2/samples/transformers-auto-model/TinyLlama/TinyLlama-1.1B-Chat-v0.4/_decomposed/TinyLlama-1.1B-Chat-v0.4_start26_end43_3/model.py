import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor):
        tmp_1 = torch.cat((in_1, in_1), dim = -1);  in_1 = None
        tmp_2 = tmp_1.cos()
        tmp_3 = tmp_2 * 1.0;  tmp_2 = None
        tmp_4 = tmp_1.sin();  tmp_1 = None
        tmp_5 = tmp_4 * 1.0;  tmp_4 = None
        tmp_6 = tmp_3.to(dtype = torch.float32);  tmp_3 = None
        tmp_7 = tmp_5.to(dtype = torch.float32);  tmp_5 = None
        _log_api_usage_once = torch._C._log_api_usage_once('python.nn_module');  _log_api_usage_once = None
        tmp_10 = in_2.to(torch.float32);  in_2 = None
        tmp_11 = tmp_10.pow(2)
        tmp_12 = tmp_11.mean(-1, keepdim = True);  tmp_11 = None
        tmp_13 = tmp_12 + 1e-05;  tmp_12 = None
        tmp_14 = torch.rsqrt(tmp_13);  tmp_13 = None
        tmp_15 = tmp_10 * tmp_14;  tmp_10 = tmp_14 = None
        tmp_16 = tmp_15.to(torch.float32);  tmp_15 = None
        tmp_17 = in_0 * tmp_16;  in_0 = tmp_16 = None
        return (tmp_6, tmp_17, tmp_7)
        