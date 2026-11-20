"""Compare the C++ loss and gradients with PyTorch autograd (float64).

    ./check --dump check.bin
    python tools/compare_with_torch.py check.bin

The weights are read from the dump in the same flat layout as model.cc
(layout()), the model is rebuilt with torch operations, and the loss and
every gradient tensor are compared.
"""
import struct
import sys

import torch
import torch.nn.functional as F

MAX_LEN = 256


def read(path):
    with open(path, "rb") as f:
        V, C, H, Fd, L, B, T, n = struct.unpack("8i", f.read(32))
        tok = torch.frombuffer(bytearray(f.read(4 * B * T)), dtype=torch.int32).long()
        tgt = torch.frombuffer(bytearray(f.read(4 * B * T)), dtype=torch.int32).long()
        w = torch.frombuffer(bytearray(f.read(4 * n)), dtype=torch.float32).double()
        g = torch.frombuffer(bytearray(f.read(4 * n)), dtype=torch.float32).double()
        loss = struct.unpack("d", f.read(8))[0]
    return dict(V=V, C=C, H=H, F=Fd, L=L, B=B, T=T), tok.view(B, T), tgt.view(B, T), w, g, loss


def split(flat, d):
    """Same order as layout() in model.cc."""
    V, C, Fd = d["V"], d["C"], d["F"]
    shapes = [("E", (V, C)), ("P", (MAX_LEN, C))]
    for l in range(d["L"]):
        shapes += [(f"{l}.ln1_g", (C,)), (f"{l}.ln1_b", (C,)), (f"{l}.Wqkv", (3 * C, C)),
                   (f"{l}.Wo", (C, C)), (f"{l}.ln2_g", (C,)), (f"{l}.ln2_b", (C,)),
                   (f"{l}.W1", (Fd, C)), (f"{l}.b1", (Fd,)), (f"{l}.W2", (C, Fd)),
                   (f"{l}.b2", (C,))]
    shapes += [("lnf_g", (C,)), ("lnf_b", (C,)), ("Wcls", (V, C)), ("bcls", (V,))]
    out, off = {}, 0
    for name, shp in shapes:
        n = 1
        for s in shp:
            n *= s
        out[name] = flat[off:off + n].view(shp)
        off += n
    assert off == flat.numel()
    return out


def forward(p, d, tok, tgt):
    B, T, C, H = d["B"], d["T"], d["C"], d["H"]
    x = p["E"][tok] + p["P"][:T]
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    for l in range(d["L"]):
        h = F.layer_norm(x, (C,), p[f"{l}.ln1_g"], p[f"{l}.ln1_b"], 1e-5)
        q, k, v = (h @ p[f"{l}.Wqkv"].T).split(C, dim=-1)
        sh = lambda t: t.view(B, T, H, C // H).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask)
        x = x + a.transpose(1, 2).reshape(B, T, C) @ p[f"{l}.Wo"].T
        h = F.layer_norm(x, (C,), p[f"{l}.ln2_g"], p[f"{l}.ln2_b"], 1e-5)
        x = x + F.relu(h @ p[f"{l}.W1"].T + p[f"{l}.b1"]) @ p[f"{l}.W2"].T + p[f"{l}.b2"]
    x = F.layer_norm(x, (C,), p["lnf_g"], p["lnf_b"], 1e-5)
    logits = x @ p["Wcls"].T + p["bcls"]
    m = tgt >= 0
    return F.cross_entropy(logits[m], tgt[m])


def main():
    d, tok, tgt, w, g_cpp, loss_cpp = read(sys.argv[1])
    w = w.clone().requires_grad_(True)
    loss = forward(split(w, d), d, tok, tgt)
    loss.backward()
    print(f"loss: C++ {loss_cpp:.8f}  torch {loss.item():.8f}")
    g_ref, g_c = split(w.grad, d), split(g_cpp, d)
    worst = 0.0
    for name in g_ref:
        r, c = g_ref[name], g_c[name]
        err = ((r - c).abs().max() / r.abs().max().clamp_min(1e-12)).item()
        worst = max(worst, err)
        print(f"  {name:8s} max|grad| {r.abs().max():.3e}  rel.err {err:.2e}")
    ok = abs(loss.item() - loss_cpp) < 1e-5 and worst < 1e-3
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
