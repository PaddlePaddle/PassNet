import torch

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, in_2 : torch.Tensor, in_3 : torch.Tensor):
        tmp_2 = torch.cat((in_2, in_2), dim = -1);  in_2 = None
        tmp_3 = tmp_2.cos()
        tmp_4 = tmp_3 * 1.0;  tmp_3 = None
        tmp_5 = tmp_2.sin();  tmp_2 = None
        tmp_6 = tmp_5 * 1.0;  tmp_5 = None
        tmp_7 = tmp_4.to(dtype = torch.float16);  tmp_4 = None
        tmp_8 = tmp_6.to(dtype = torch.float16);  tmp_6 = None
        _log_api_usage_once = torch._C._log_api_usage_once('python.nn_module');  _log_api_usage_once = None
        tmp_11 = torch.nn.functional.layer_norm(in_3, (2048,), in_1, in_0, 1e-05);  in_3 = in_1 = in_0 = None
        return (tmp_7, tmp_11, tmp_8)
        