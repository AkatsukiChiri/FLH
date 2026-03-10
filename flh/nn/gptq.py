import math
import copy
import torch
import torch.nn as nn

from .quantization import fast_hadamard_transform


def _sym_quantize(x, scale, maxq, clip_val=None):
    scale = scale.to(x.device)
    if clip_val is not None:
        x = torch.clamp(x, -clip_val, clip_val)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return scale * q


def _asym_quantize(x, scale, zero, maxq, clip_val=None):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    if clip_val is not None:
        x = torch.clamp(x, -clip_val, clip_val)
    q = torch.clamp(torch.round(x / scale + zero), 0, maxq)
    return scale * (q - zero)


class GPTQQuantizer:
    def __init__(self, bits, groupsize=-1, sym=True, clip_ratio=1.0):
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        self.maxq = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 1
        self.scale = None
        self.zero = None

    def ready(self):
        return self.scale is not None and torch.all(self.scale != 0)

    def _clip_weight(self, W):
        if self.clip_ratio >= 1.0:
            return W
        if self.groupsize > 0 and W.shape[1] % self.groupsize == 0:
            Wg = W.view(W.shape[0], -1, self.groupsize)
            max_abs = Wg.abs().amax(dim=2, keepdim=True)
            clip_val = max_abs * self.clip_ratio
            return torch.clamp(Wg, -clip_val, clip_val).view(W.shape)
        max_abs = W.abs().amax(dim=1, keepdim=True)
        clip_val = max_abs * self.clip_ratio
        return torch.clamp(W, -clip_val, clip_val)

    def find_params(self, W, weight=True):
        W = W.float()
        W = self._clip_weight(W)
        if self.bits >= 16:
            return
        dev = W.device
        self.maxq = torch.tensor(self.maxq, device=dev)
        shape = W.shape

        if self.groupsize > 0 and W.shape[1] % self.groupsize == 0:
            # Per-group
            W_grouped = W.view(W.shape[0], -1, self.groupsize)
            if self.sym:
                xmax = W_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-5)
                self.scale = (xmax / self.maxq).view(W.shape[0], -1)
                self.zero = torch.zeros_like(self.scale)
            else:
                xmax = W_grouped.amax(dim=2, keepdim=True)
                xmin = W_grouped.amin(dim=2, keepdim=True)
                tmp = (xmin == 0) & (xmax == 0)
                xmin = torch.where(tmp, torch.tensor(-1.0, device=dev), xmin)
                xmax = torch.where(tmp, torch.tensor(1.0, device=dev), xmax)
                self.scale = ((xmax - xmin) / self.maxq).clamp(min=1e-5).view(W.shape[0], -1)
                self.zero = torch.round(-xmin / self.scale).view(W.shape[0], -1)
        else:
            # Per-channel
            W_flat = W.flatten(1)
            xmin = W_flat.min(1)[0]
            xmax = W_flat.max(1)[0]
            if self.sym:
                xmax = torch.maximum(xmax.abs(), xmin.abs()).clamp(min=1e-5)
                self.scale = (xmax / self.maxq).unsqueeze(1)
                self.zero = torch.zeros_like(self.scale)
            else:
                tmp = (xmin == 0) & (xmax == 0)
                xmin = torch.where(tmp, torch.tensor(-1.0, device=dev), xmin)
                xmax = torch.where(tmp, torch.tensor(1.0, device=dev), xmax)
                self.scale = ((xmax - xmin) / self.maxq).clamp(min=1e-5).unsqueeze(1)
                self.zero = torch.round(-xmin / self.scale).unsqueeze(1)
            self.scale = self.scale.repeat(1, W.shape[1])
            self.zero = self.zero.repeat(1, W.shape[1])

    def quantize(self, w, col=0):
        if not self.ready():
            return w
        if self.scale.dim() == 1:
            scale = self.scale
        else:
            scale = self.scale[:, col] if col < self.scale.shape[1] else self.scale[:, 0]
        zero = None
        if not self.sym and self.zero is not None:
            zero = self.zero[:, col] if self.zero.dim() > 1 and col < self.zero.shape[1] else self.zero[:, 0]
        clip_val = None
        if self.clip_ratio < 1.0:
            clip_val = scale * self.maxq * self.clip_ratio
            clip_val = clip_val.to(w.device)
        if self.sym:
            return _sym_quantize(w, scale, self.maxq, clip_val)
        return _asym_quantize(w, scale, zero, self.maxq, clip_val)


class FLHGPTQ:
    def __init__(self, layer, dual_hadamard=False):
        self.layer = layer
        self.dual_hadamard = dual_hadamard
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0
        self.out_list = []

    def add_batch(self, inp, out, group_size=-1):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0] * inp.shape[1] if len(inp.shape) == 3 else inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1])
        inp = fast_hadamard_transform(inp.float(), group_size=group_size if group_size > 0 else None)
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H += inp.matmul(inp.t())
        if out is not None:
            if len(out.shape) == 2:
                out = out.unsqueeze(0)
            out_flat = out.reshape(-1, out.shape[-1])
            self.out_list.append(out_flat)

    def fasterquant(
        self,
        group_size=128,
        blocksize=128,
        percdamp=0.01,
        actorder=False,
        sym=True,
        bits=4,
        dual_hadamard=None,
        clip_ratio=1.0,
    ):
        if dual_hadamard is None:
            dual_hadamard = self.dual_hadamard
        W = self.layer.weight.data.clone().float()
        group_size = group_size if group_size > 0 and self.columns % group_size == 0 else -1
        gs = group_size if group_size > 0 else None
        
        if dual_hadamard:
            W = fast_hadamard_transform(W, group_size=gs)
            W = fast_hadamard_transform(W.T, group_size=gs).T
        else:
            W = fast_hadamard_transform(W, group_size=gs)

        self.quantizer = GPTQQuantizer(bits=bits, groupsize=group_size, sym=sym, clip_ratio=clip_ratio)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        static_groups = group_size > 0
        groups = []
        if clip_ratio < 1.0:
            W = self.quantizer._clip_weight(W)
        if static_groups:
            for i in range(0, self.columns, group_size):
                q = copy.deepcopy(self.quantizer)
                q.find_params(W[:, i : (i + group_size)])
                groups.append(q)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Q = torch.zeros_like(W)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                col_idx = (i1 + i) % group_size if group_size > 0 else 0

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + group_size)])
                        q = self.quantizer.quantize(w, col_idx)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx].item()
                        q = groups[idx // group_size].quantize(w, col_idx)
                else:
                    if not self.quantizer.ready():
                        self.quantizer.find_params(W)
                    q = self.quantizer.quantize(w, 0)

                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.to(self.layer.weight.dtype).reshape(self.layer.weight.shape)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
