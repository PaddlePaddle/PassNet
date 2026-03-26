import torch
from torch import device

class GraphModule(torch.nn.Module):

    def forward(self, in_0, in_1, in_2, in_3, in_4):
        tmp_0 = in_0
        tmp_1 = in_1
        tmp_2 = in_2
        tmp_3 = in_3
        tmp_4 = in_4
        tmp_5 = torch.nn.functional.embedding(tmp_1, tmp_2, None, None, 2.0, False, False)
        tmp_1 = tmp_2 = None
        tmp_6 = torch.arange(0, 512, device=device(type='cuda', index=0))
        tmp_7 = tmp_6.unsqueeze(0)
        tmp_8 = tmp_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
        tmp_0 = None
        tmp_9 = torch.arange(512, device=device(type='cuda', index=0))
        tmp_9 += 0
        tmp_10 = tmp_9
        tmp_9 = None
        tmp_11 = tmp_8[slice(None, None, None), tmp_10]
        tmp_8 = tmp_10 = None
        tmp_12 = torch.arange(512, device=device(type='cuda', index=0))
        tmp_12 += 0
        tmp_13 = tmp_12
        tmp_12 = None
        tmp_14 = tmp_6.view(-1, 1)
        tmp_6 = None
        tmp_15 = tmp_13 <= tmp_14
        tmp_13 = tmp_14 = None
        tmp_16 = tmp_15[None, None, slice(None, None, None), slice(None, None, None)]
        tmp_15 = None
        tmp_17 = tmp_16.expand(1, -1, -1, -1)
        tmp_16 = None
        tmp_18 = tmp_11[slice(None, None, None), None, None, slice(None, None, None)]
        tmp_11 = None
        tmp_19 = tmp_17 * tmp_18
        tmp_17 = tmp_18 = None
        tmp_20 = torch.set_grad_enabled(False)
        tmp_20 = None
        tmp_21 = tmp_4[None, slice(None, None, None), None]
        tmp_4 = None
        tmp_22 = tmp_21.float()
        tmp_21 = None
        tmp_23 = tmp_22.expand(1, -1, 1)
        tmp_22 = None
        tmp_24 = tmp_23.to(device(type='cuda', index=0))
        tmp_23 = None
        tmp_25 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
        tmp_7 = None
        tmp_26 = tmp_25.float()
        tmp_25 = None
        tmp_27 = tmp_24.float()
        tmp_24 = None
        tmp_28 = tmp_26.float()
        tmp_26 = None
        tmp_29 = tmp_27 @ tmp_28
        tmp_27 = tmp_28 = None
        tmp_30 = tmp_29.transpose(1, 2)
        tmp_29 = None
        tmp_31 = torch.cat((tmp_30, tmp_30), dim=-1)
        tmp_30 = None
        tmp_32 = tmp_31.cos()
        tmp_33 = tmp_32 * 1.0
        tmp_32 = None
        tmp_34 = tmp_31.sin()
        tmp_31 = None
        tmp_35 = tmp_34 * 1.0
        tmp_34 = None
        tmp_36 = tmp_33.to(dtype=torch.float32)
        tmp_33 = None
        tmp_37 = tmp_35.to(dtype=torch.float32)
        tmp_35 = None
        tmp_38 = torch.set_grad_enabled(True)
        tmp_38 = None
        tmp_39 = torch._C._log_api_usage_once('python.nn_module')
        tmp_39 = None
        tmp_40 = tmp_5.to(torch.float32)
        tmp_41 = tmp_40.pow(2)
        tmp_42 = tmp_41.mean(-1, keepdim=True)
        tmp_41 = None
        tmp_43 = tmp_42 + 1e-05
        tmp_42 = None
        tmp_44 = torch.rsqrt(tmp_43)
        tmp_43 = None
        tmp_45 = tmp_40 * tmp_44
        tmp_40 = tmp_44 = None
        tmp_46 = tmp_45.to(torch.float32)
        tmp_45 = None
        tmp_47 = tmp_3 * tmp_46
        tmp_3 = tmp_46 = None
        return (tmp_19, tmp_36, tmp_47, tmp_5, tmp_37)