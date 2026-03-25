import torch

from torch import device

class GraphModule(torch.nn.Module):
    
    
    
    def forward(self, in_0 : torch.Tensor, in_1 : torch.Tensor, w_0 : torch.Tensor, w_1 : torch.Tensor):
        tmp_4 = torch.arange(0, 2, device = device(type='cuda', index=0))
        tmp_5 = tmp_4.unsqueeze(0)
        tmp_6 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool);  in_0 = None
        tmp_7 = torch.arange(2, device = device(type='cuda', index=0))
        tmp_7 += 0;  tmp_8 = tmp_7;  tmp_7 = None
        tmp_9 = tmp_6[(slice(None, None, None), tmp_8)];  tmp_6 = tmp_8 = None
        tmp_10 = torch.arange(2, device = device(type='cuda', index=0))
        tmp_10 += 0;  tmp_11 = tmp_10;  tmp_10 = None
        tmp_12 = tmp_4.view(-1, 1);  tmp_4 = None
        tmp_13 = tmp_11 <= tmp_12;  tmp_11 = tmp_12 = None
        tmp_14 = tmp_13[(None, None, slice(None, None, None), slice(None, None, None))];  tmp_13 = None
        tmp_15 = tmp_14.expand(1, -1, -1, -1);  tmp_14 = None
        tmp_16 = tmp_9[(slice(None, None, None), None, None, slice(None, None, None))];  tmp_9 = None
        tmp_17 = tmp_15 * tmp_16;  tmp_15 = tmp_16 = None
        _set_grad_enabled = torch.set_grad_enabled(False);  _set_grad_enabled = None
        tmp_19 = w_1[(None, slice(None, None, None), None)];  w_1 = None
        tmp_20 = tmp_19.float();  tmp_19 = None
        tmp_21 = tmp_20.expand(1, -1, 1);  tmp_20 = None
        tmp_22 = tmp_21.to(device(type='cuda', index=0));  tmp_21 = None
        tmp_23 = tmp_5[(slice(None, None, None), None, slice(None, None, None))];  tmp_5 = None
        tmp_24 = tmp_23.float();  tmp_23 = None
        tmp_25 = tmp_22.float();  tmp_22 = None
        tmp_26 = tmp_24.float();  tmp_24 = None
        to = tmp_25.to(torch.float16);  tmp_25 = None
        to_1 = tmp_26.to(torch.float16);  tmp_26 = None
        matmul = to @ to_1;  to = to_1 = None
        tmp_28 = matmul.transpose(1, 2);  matmul = None
        tmp_29 = torch.cat((tmp_28, tmp_28), dim = -1);  tmp_28 = None
        tmp_30 = tmp_29.cos()
        tmp_31 = tmp_30 * 1.0;  tmp_30 = None
        tmp_32 = tmp_29.sin();  tmp_29 = None
        tmp_33 = tmp_32 * 1.0;  tmp_32 = None
        _set_grad_enabled_1 = torch.set_grad_enabled(True);  _set_grad_enabled_1 = None
        _log_api_usage_once = torch._C._log_api_usage_once('python.nn_module');  _log_api_usage_once = None
        tmp_36 = in_1.to(torch.float32);  in_1 = None
        tmp_37 = tmp_36.pow(2)
        tmp_38 = tmp_37.mean(-1, keepdim = True);  tmp_37 = None
        tmp_39 = tmp_38 + 1e-05;  tmp_38 = None
        tmp_40 = torch.rsqrt(tmp_39);  tmp_39 = None
        tmp_41 = tmp_36 * tmp_40;  tmp_36 = tmp_40 = None
        tmp_42 = tmp_41.to(torch.bfloat16);  tmp_41 = None
        tmp_43 = w_0 * tmp_42;  w_0 = tmp_42 = None
        return (tmp_17, tmp_31, tmp_43, tmp_33)
        