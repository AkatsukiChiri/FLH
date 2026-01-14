import torch
# import flh

class Quantizer(torch.nn.Module):
    def __init__(self, bits=8, group_size=-1, sym=True, input_clip_ratio=1.0):
        """
        bits:        int, number of quantization bits (1~16)
        group_size:  int, number of channels per group (if -1, disable group quant)
        sym:         bool, whether to use symmetric quantization
        input_clip_ratio: float, clipping ratio for input activation
        """
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.input_clip_ratio = input_clip_ratio

    def forward(self, x):
        if self.bits >= 16:
            return x
        
        x_clipped = torch.clamp(x, -self.input_clip_ratio, self.input_clip_ratio)
        orig_shape = x_clipped.shape

        # For last-dim (usually channel/out feature dim)
        if self.group_size is not None and self.group_size > 0 and x_clipped.size(-1) % self.group_size == 0:
            # Reshape to (..., n_groups, group_size)
            n_groups = x_clipped.size(-1) // self.group_size
            new_shape = x_clipped.shape[:-1] + (n_groups, self.group_size)
            x_grouped = x_clipped.view(new_shape)

            if self.sym:
                max_val = x_grouped.abs().amax(dim=-1, keepdim=True)
                min_val = -max_val
            else:
                max_val = x_grouped.amax(dim=-1, keepdim=True)
                min_val = x_grouped.amin(dim=-1, keepdim=True)

            qmin = 0
            qmax = 2 ** self.bits - 1

            # scale/zero, (for symmetric do centralize, else stay offset)
            scale = (max_val - min_val) / (qmax - qmin + 1e-8)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero_point = -min_val / scale if not self.sym else torch.zeros_like(scale)

            x_int = ((x_grouped - min_val) / scale).round() + qmin if not self.sym else ((x_grouped / scale).round()).clip(qmin, qmax)
            x_deq = x_int * scale + (min_val if not self.sym else 0)
            x_deq = x_deq.view(orig_shape)
        else:
            # Per-tensor or per-channel quantization
            if self.sym:
                max_val = x_clipped.abs().amax(dim=-1, keepdim=True)
                min_val = -max_val
            else:
                max_val = x_clipped.amax(dim=-1, keepdim=True)
                min_val = x_clipped.amin(dim=-1, keepdim=True)
            
            qmin = 0
            qmax = 2 ** self.bits - 1
            
            scale = (max_val - min_val) / (qmax - qmin + 1e-8)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            zero_point = -min_val / scale if not self.sym else torch.zeros_like(scale)

            if self.sym:
                x_int = ((x_clipped / scale).round()).clip(qmin, qmax)
                x_deq = x_int * scale
            else:
                x_int = ((x_clipped - min_val) / scale).round() + qmin
                x_int = x_int.clip(qmin, qmax)
                x_deq = x_int * scale + min_val 
        return x_deq
