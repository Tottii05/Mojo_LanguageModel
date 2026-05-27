from std.collections import List, Dict
from std.math import exp, log, sqrt
from std.random import random_float64, seed
from std.pathlib import Path
from std.python import Python
from std.memory import UnsafePointer, memset_zero, alloc

comptime F64 = DType.float64
comptime SIMD_WIDTH: Int = 8

def simd_dot(a: UnsafePointer[Scalar[F64], _], b: UnsafePointer[Scalar[F64], _], length: Int) -> Float64:
    var acc = SIMD[F64, SIMD_WIDTH](0)
    var i = 0
    while i + SIMD_WIDTH <= length:
        acc += a.load[width=SIMD_WIDTH](i) * b.load[width=SIMD_WIDTH](i)
        i += SIMD_WIDTH
    var tail: Float64 = 0.0
    for j in range(i, length):
        tail += a[j] * b[j]
    for k in range(SIMD_WIDTH):
        tail += acc[k]
    return tail

def rand_matrix_scaled(rows: Int, cols: Int, scale: Float64) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    var n = rows * cols
    var ptr = alloc[Scalar[F64]](n)
    for i in range(n):
        ptr[i] = (random_float64() - 0.5) * scale
    return ptr

def zeros(n: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    var ptr = alloc[Scalar[F64]](n)
    memset_zero(ptr, n)
    return ptr

def zeros_matrix(rows: Int, cols: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    var ptr = alloc[Scalar[F64]](rows * cols)
    memset_zero(ptr, rows * cols)
    return ptr

def linear[origin_x: Origin, origin_w: Origin](x: UnsafePointer[Scalar[F64], origin_x], w: UnsafePointer[Scalar[F64], origin_w], out_dim: Int, in_dim: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    var out = zeros(out_dim)
    for i in range(out_dim):
        var w_row = w + i * in_dim
        out[i] = Scalar[F64](simd_dot(x, w_row, in_dim))
    return out

def relu[origin: Origin](x: UnsafePointer[Scalar[F64], origin], length: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    var out = zeros(length)
    for i in range(length):
        var val = x[i]
        out[i] = val if val > 0.0 else Scalar[F64](0)
    return out

def relu_backward[origin_go: Origin, origin_x: Origin](grad_out: UnsafePointer[Scalar[F64], origin_go], x: UnsafePointer[Scalar[F64], origin_x], length: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    var grad = zeros(length)
    for i in range(length):
        grad[i] = grad_out[i] if x[i] > 0.0 else Scalar[F64](0)
    return grad

def softmax[origin: Origin](logits: UnsafePointer[Scalar[F64], origin], length: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    if length == 0:
        return zeros(0)

    var maxv = logits[0]
    for i in range(1, length):
        if logits[i] > maxv:
            maxv = logits[i]

    var exps = zeros(length)
    var sumv: Float64 = 0.0
    for i in range(length):
        var e = exp(logits[i] - maxv)
        exps[i] = e
        sumv += e

    var out = zeros(length)
    for i in range(length):
        out[i] = exps[i] / sumv
    return out

def softmax_attn[origin: Origin](scores: UnsafePointer[Scalar[F64], origin], seq_len: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    if seq_len == 0:
        return zeros(0)
    
    var maxv = scores[0]
    for i in range(1, seq_len):
        if scores[i] > maxv:
            maxv = scores[i]

    var exps = zeros(seq_len)
    var sumv: Float64 = 0.0

    for i in range(seq_len):
        var e = exp(scores[i] - maxv)
        exps[i] = e
        sumv += e

    var out = zeros(seq_len)
    for i in range(seq_len):
        out[i] = exps[i] / sumv
    return out

struct GradW:
    var gw: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var gx: UnsafePointer[Scalar[F64], MutExternalOrigin]
    def __init__(out self, gw: UnsafePointer[Scalar[F64], MutExternalOrigin], gx: UnsafePointer[Scalar[F64], MutExternalOrigin]):
        self.gw = gw
        self.gx = gx

def linear_backward[origin_go: Origin, origin_x: Origin, origin_w: Origin](grad_out: UnsafePointer[Scalar[F64], origin_go], x: UnsafePointer[Scalar[F64], origin_x], w: UnsafePointer[Scalar[F64], origin_w], out_dim: Int, in_dim: Int) -> GradW:
    var gw = zeros(out_dim * in_dim)
    var gx = zeros(in_dim)

    for i in range(out_dim):
        var go = grad_out[i]
        var w_offset = i * in_dim
        var j = 0
        while j + SIMD_WIDTH <= in_dim:
            var vx = x.load[width=SIMD_WIDTH](j)
            var contrib = go * vx
            gw.store[width=SIMD_WIDTH](w_offset + j, gw.load[width=SIMD_WIDTH](w_offset + j) + contrib)
            j += SIMD_WIDTH

        while j < in_dim:
            var contrib = go * x[j]
            gw[w_offset + j] = contrib
            j += 1

        for j in range(in_dim):
            gx[j] += go * w[w_offset + j]

    return GradW(gw, gx)

struct AttnCache:
    var q: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var k: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var v: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var scores: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var attn_w: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var attn_out: UnsafePointer[Scalar[F64], MutExternalOrigin]

    def __init__(out self, q: UnsafePointer[Scalar[F64], MutExternalOrigin], k: UnsafePointer[Scalar[F64], MutExternalOrigin], v: UnsafePointer[Scalar[F64], MutExternalOrigin], scores: UnsafePointer[Scalar[F64], MutExternalOrigin], attn_w: UnsafePointer[Scalar[F64], MutExternalOrigin], attn_out: UnsafePointer[Scalar[F64], MutExternalOrigin]):
        self.q = q
        self.k = k
        self.v = v
        self.scores = scores
        self.attn_w = attn_w
        self.attn_out = attn_out

def attn_head_forward[origin_xs: Origin, origin_Wq: Origin, origin_Wk: Origin, origin_Wv: Origin](xs: UnsafePointer[Scalar[F64], origin_xs], Wq: UnsafePointer[Scalar[F64], origin_Wq], Wk: UnsafePointer[Scalar[F64], origin_Wk], Wv: UnsafePointer[Scalar[F64], origin_Wv], seq_len: Int, embd: Int, head_dim: Int) -> AttnCache:
    var q = zeros(seq_len * head_dim)
    var k = zeros(seq_len * head_dim)
    var v = zeros(seq_len * head_dim)

    for t in range(seq_len):
        var x_offset = t * embd
        var qt = linear(xs + x_offset, Wq, head_dim, embd)
        var kt = linear(xs + x_offset, Wk, head_dim, embd)
        var vt = linear(xs + x_offset, Wv, head_dim, embd)
        var q_offset = t * head_dim
        var k_offset = t * head_dim
        var v_offset = t * head_dim

        for d in range(head_dim):
            q[q_offset + d] = qt[d]
            k[k_offset + d] = kt[d]
            v[v_offset + d] = vt[d]

    var scale = 1.0 / (Float64(head_dim) ** 0.5)
    var scores = zeros(seq_len * seq_len)
    var attn_w = zeros(seq_len * seq_len)

    for t in range(seq_len):
        var t_offset = t * seq_len
        for s in range(seq_len):
            if s > t:
                scores[t_offset + s] = -1e9
                continue

            var dot: Float64 = 0.0
            for d in range(head_dim):
                dot += q[t * head_dim + d] * k[s * head_dim + d]
            
            scores[t_offset + s] = dot * scale
        var row_ptr = scores + t_offset
        var attn_row = softmax_attn(row_ptr, seq_len)
        
        for s in range(seq_len):
            attn_w[t_offset + s] = attn_row[s]

    var attn_out = zeros(seq_len * head_dim)

    for t in range(seq_len):
        var t_offset = t * seq_len
        var out_offset = t * head_dim
        
        for d in range(head_dim):
            var acc: Float64 = 0.0
            for s in range(seq_len):
                acc += attn_w[t_offset + s] * v[s * head_dim + d]
            attn_out[out_offset + d] = acc

    return AttnCache(q, k, v, scores, attn_w, attn_out)

struct AttnGrad:
    var d_xs: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wq: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wk: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wv: UnsafePointer[Scalar[F64], MutExternalOrigin]

    def __init__(out self, d_xs: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wq: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wk: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wv: UnsafePointer[Scalar[F64], MutExternalOrigin]):
        self.d_xs = d_xs
        self.g_Wq = g_Wq
        self.g_Wk = g_Wk
        self.g_Wv = g_Wv

def attn_head_backward[origin_dout: Origin, origin_xs: Origin, origin_Wq: Origin,origin_Wk: Origin, origin_Wv: Origin](d_out: UnsafePointer[Scalar[F64], origin_dout], cache: AttnCache, xs: UnsafePointer[Scalar[F64], origin_xs], Wq: UnsafePointer[Scalar[F64], origin_Wq], Wk: UnsafePointer[Scalar[F64], origin_Wk], Wv: UnsafePointer[Scalar[F64], origin_Wv], seq_len: Int, embd: Int, head_dim: Int) -> AttnGrad:
    var scale = 1.0 / (Float64(head_dim) ** 0.5)
    var d_v = zeros(seq_len * head_dim)
    var d_attnw = zeros(seq_len * seq_len)

    for t in range(seq_len):
        var t_out_offset = t * head_dim
        var t_attn_offset = t * seq_len
        for s in range(seq_len):
            for d in range(head_dim):
                d_v[s * head_dim + d] += cache.attn_w[t_attn_offset + s] * d_out[t_out_offset + d]
                d_attnw[t_attn_offset + s] += d_out[t_out_offset + d] * cache.v[s * head_dim + d]

    var d_scores = zeros(seq_len * seq_len)
    for t in range(seq_len):
        var t_offset = t * seq_len
        var dot: Float64 = 0.0
        for s in range(seq_len):
            dot += cache.attn_w[t_offset + s] * d_attnw[t_offset + s]

        for s in range(seq_len):
            if s <= t:
                var aw = cache.attn_w[t_offset + s]
                d_scores[t_offset + s] = aw * (d_attnw[t_offset + s] - dot)

    var d_q = zeros(seq_len * head_dim)
    var d_k = zeros(seq_len * head_dim)

    for t in range(seq_len):
        var t_offset = t * seq_len
        for s in range(seq_len):
            if s <= t:
                var ds = d_scores[t_offset + s] * scale
                for d in range(head_dim):
                    d_q[t * head_dim + d] += ds * cache.k[s * head_dim + d]
                    d_k[s * head_dim + d] += ds * cache.q[t * head_dim + d]

    var g_Wq = zeros(head_dim * embd)
    var g_Wk = zeros(head_dim * embd)
    var g_Wv = zeros(head_dim * embd)
    var d_xs = zeros(seq_len * embd)

    for t in range(seq_len):
        var x_offset = t * embd
        var dq_offset = t * head_dim
        var dk_offset = t * head_dim
        var dv_offset = t * head_dim
        var bq = linear_backward(d_q + dq_offset, xs + x_offset, Wq, head_dim, embd)
        var bk = linear_backward(d_k + dk_offset, xs + x_offset, Wk, head_dim, embd)
        var bv = linear_backward(d_v + dv_offset, xs + x_offset, Wv, head_dim, embd)

        for i in range(head_dim * embd):
            g_Wq[i] += bq.gw[i]
            g_Wk[i] += bk.gw[i]
            g_Wv[i] += bv.gw[i]

        for j in range(embd):
            d_xs[x_offset + j] += bq.gx[j] + bk.gx[j] + bv.gx[j]
    return AttnGrad(d_xs, g_Wq, g_Wk, g_Wv)

struct StepResult:
    var loss: Float64
    var g_Wout: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_W1: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wq1: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wk1: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wv1: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wq2: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wk2: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wv2: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var g_Wte: UnsafePointer[Scalar[F64], MutExternalOrigin]
    def __init__(out self, loss: Float64, g_Wout: UnsafePointer[Scalar[F64], MutExternalOrigin], g_W1: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wq1: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wk1: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wv1: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wq2: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wk2: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wv2: UnsafePointer[Scalar[F64], MutExternalOrigin], g_Wte: UnsafePointer[Scalar[F64], MutExternalOrigin]):
        self.loss = loss
        self.g_Wout = g_Wout
        self.g_W1   = g_W1
        self.g_Wq1  = g_Wq1
        self.g_Wk1  = g_Wk1
        self.g_Wv1  = g_Wv1
        self.g_Wq2  = g_Wq2
        self.g_Wk2  = g_Wk2
        self.g_Wv2  = g_Wv2
        self.g_Wte  = g_Wte

struct AdamResult:
    var w: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var m: UnsafePointer[Scalar[F64], MutExternalOrigin]
    var v: UnsafePointer[Scalar[F64], MutExternalOrigin]
    def __init__(out self, w: UnsafePointer[Scalar[F64], MutExternalOrigin], m: UnsafePointer[Scalar[F64], MutExternalOrigin], v: UnsafePointer[Scalar[F64], MutExternalOrigin]):
        self.w = w
        self.m = m
        self.v = v

def adam_update[origin_w: Origin, origin_g: Origin, origin_m: Origin, origin_v: Origin](w: UnsafePointer[Scalar[F64], origin_w], grad: UnsafePointer[Scalar[F64], origin_g], m: UnsafePointer[Scalar[F64], origin_m], v: UnsafePointer[Scalar[F64], origin_v], length: Int, t: Int, lr: Float64, beta1: Float64, beta2: Float64, eps: Float64) -> AdamResult:
    var w_out = zeros(length)
    var m_out = zeros(length)
    var v_out = zeros(length)
    var ft = Float64(t)
    var bc1 = 1.0 - beta1 ** ft
    var bc2 = 1.0 - beta2 ** ft

    for i in range(length):
        var mi = beta1 * m[i] + (1.0 - beta1) * grad[i]
        var vi = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i]
        var mh = mi / bc1
        var vh = vi / bc2
        w_out[i] = w[i] - lr * mh / (vh ** 0.5 + eps)
        m_out[i] = mi
        v_out[i] = vi
    return AdamResult(w_out, m_out, v_out)

def train_step[origin_tokens: Origin, origin_targets: Origin, origin_Wte: Origin, origin_Wout: Origin, origin_W1: Origin, origin_Wq1: Origin, origin_Wk1: Origin, origin_Wv1: Origin, origin_Wq2: Origin, origin_Wk2: Origin, origin_Wv2: Origin]
    (tokens: UnsafePointer[Scalar[F64], origin_tokens], targets: UnsafePointer[Scalar[F64], origin_targets], seq_len: Int, Wte: UnsafePointer[Scalar[F64], origin_Wte], Wout: UnsafePointer[Scalar[F64], origin_Wout], W1: UnsafePointer[Scalar[F64], origin_W1], Wq1: UnsafePointer[Scalar[F64], origin_Wq1], Wk1: UnsafePointer[Scalar[F64], origin_Wk1], Wv1: UnsafePointer[Scalar[F64], origin_Wv1], Wq2: UnsafePointer[Scalar[F64], origin_Wq2], Wk2: UnsafePointer[Scalar[F64], origin_Wk2], Wv2: UnsafePointer[Scalar[F64], origin_Wv2], lr: Float64, vocab: Int, embd: Int, hidden: Int, head_dim: Int) -> StepResult:
    var attn_dim = 2 * head_dim
    # Forward: Embeddings
    var xs = zeros(seq_len * embd)
    for t in range(seq_len):
        var tid = Int(tokens[t])
        var emb_offset = tid * embd
        var x_offset = t * embd
        for j in range(embd):
            xs[x_offset + j] = Wte[emb_offset + j]

    # Forward: Attention heads
    var c1 = attn_head_forward(xs, Wq1, Wk1, Wv1, seq_len, embd, head_dim)
    var c2 = attn_head_forward(xs, Wq2, Wk2, Wv2, seq_len, embd, head_dim)
    var attn_out = zeros(seq_len * attn_dim)
    for t in range(seq_len):
        var out_offset = t * attn_dim
        var c1_offset = t * head_dim
        var c2_offset = t * head_dim
        for d in range(head_dim):
            attn_out[out_offset + d] = c1.attn_out[c1_offset + d]
            attn_out[out_offset + head_dim + d] = c2.attn_out[c2_offset + d]

    # Forward: MLP + logits
    var h_pres = zeros(seq_len * hidden)
    var hs = zeros(seq_len * hidden)
    var probs_all = zeros(seq_len * vocab)

    for t in range(seq_len):
        var at_offset = t * attn_dim
        var h_pre = linear(attn_out + at_offset, W1, hidden, attn_dim)
        var h = relu(h_pre, hidden)
        var logits = linear(h, Wout, vocab, hidden)
        var probs = softmax(logits, vocab)
        var h_offset = t * hidden
        for d in range(hidden):
            h_pres[h_offset + d] = h_pre[d]
            hs[h_offset + d] = h[d]

        var p_offset = t * vocab
        for v in range(vocab):
            probs_all[p_offset + v] = probs[v]
    # Loss
    var total_loss: Float64 = 0.0
    for t in range(seq_len):
        var tgt = Int(targets[t])
        var p = probs_all[t * vocab + tgt]
        if p < 1e-8:
            p = 1e-8
        total_loss += -log(p)
    
    total_loss = total_loss / Float64(seq_len)

    # Backward
    var g_Wout = zeros(vocab * hidden)
    var g_W1   = zeros(hidden * attn_dim)
    var d_attn = zeros(seq_len * attn_dim)

    for t in range(seq_len):
        var tgt = Int(targets[t])
        var p_offset = t * vocab
        var d_logits = zeros(vocab)
        for v in range(vocab):
            var g = probs_all[p_offset + v]
            if v == tgt:
                g = g - 1.0
            d_logits[v] = g

        var h_offset = t * hidden
        var bk_out = linear_backward(d_logits, hs + h_offset, Wout, vocab, hidden)

        for i in range(vocab * hidden):
            g_Wout[i] += bk_out.gw[i]

        var h_pre_offset = t * hidden
        var d_h_pre = relu_backward(bk_out.gx, h_pres + h_pre_offset, hidden)
        var at_offset = t * attn_dim
        var bk_W1 = linear_backward(d_h_pre, attn_out + at_offset, W1, hidden, attn_dim)

        for i in range(hidden * attn_dim):
            g_W1[i] += bk_W1.gw[i]

        for d in range(attn_dim):
            d_attn[at_offset + d] += bk_W1.gx[d]

    var d_out1 = zeros(seq_len * head_dim)
    var d_out2 = zeros(seq_len * head_dim)

    for t in range(seq_len):
        var src_offset = t * attn_dim
        var dst_offset = t * head_dim
        for d in range(head_dim):
            d_out1[dst_offset + d] = d_attn[src_offset + d]
            d_out2[dst_offset + d] = d_attn[src_offset + head_dim + d]

    var ag1 = attn_head_backward(d_out1, c1, xs, Wq1, Wk1, Wv1, seq_len, embd, head_dim)
    var ag2 = attn_head_backward(d_out2, c2, xs, Wq2, Wk2, Wv2, seq_len, embd, head_dim)
    var g_Wte = zeros(vocab * embd)
    for t in range(seq_len):
        var tid = Int(tokens[t])
        var emb_offset = tid * embd
        var dx_offset = t * embd
        for j in range(embd):
            g_Wte[emb_offset + j] += ag1.d_xs[dx_offset + j] + ag2.d_xs[dx_offset + j]
    return StepResult(total_loss, g_Wout, g_W1, ag1.g_Wq, ag1.g_Wk, ag1.g_Wv, ag2.g_Wq, ag2.g_Wk, ag2.g_Wv, g_Wte)

def sample_from_logits[origin: Origin](logits: UnsafePointer[Scalar[F64], origin], length: Int, temperature: Float64 = 1.0) -> Int:
    if temperature < 0.01:
        var best = 0
        var best_val = logits[0]
        for i in range(1, length):
            if logits[i] > best_val:
                best_val = logits[i]
                best = i
        return best
    var probs = softmax(logits, length)
    var r = random_float64()
    var cum: Float64 = 0.0
    
    for i in range(length):
        cum += probs[i]
        if r <= cum:
            return i
    return length - 1

def model_forward[origin_tokens: Origin, origin_Wte: Origin, origin_Wout: Origin, origin_W1: Origin, origin_Wq1: Origin, origin_Wk1: Origin, origin_Wv1: Origin, origin_Wq2: Origin, origin_Wk2: Origin, origin_Wv2: Origin](
    tokens: UnsafePointer[Scalar[F64], origin_tokens],
    seq_len: Int,
    Wte: UnsafePointer[Scalar[F64], origin_Wte],
    Wout: UnsafePointer[Scalar[F64], origin_Wout],
    W1: UnsafePointer[Scalar[F64], origin_W1],
    Wq1: UnsafePointer[Scalar[F64], origin_Wq1],
    Wk1: UnsafePointer[Scalar[F64], origin_Wk1],
    Wv1: UnsafePointer[Scalar[F64], origin_Wv1],
    Wq2: UnsafePointer[Scalar[F64], origin_Wq2],
    Wk2: UnsafePointer[Scalar[F64], origin_Wk2],
    Wv2: UnsafePointer[Scalar[F64], origin_Wv2],
    vocab: Int,
    embd: Int,
    hidden: Int,
    head_dim: Int) -> UnsafePointer[Scalar[F64], MutExternalOrigin]:
    var attn_dim = 2 * head_dim
    # Forward de los embeddings
    var xs = zeros(seq_len * embd)
    for t in range(seq_len):
        var tid = Int(tokens[t])
        var emb_offset = tid * embd
        var x_offset = t * embd
        for j in range(embd):
            xs[x_offset + j] = Wte[emb_offset + j]
    # Forward de la atencion
    var c1 = attn_head_forward(xs, Wq1, Wk1, Wv1, seq_len, embd, head_dim)
    var c2 = attn_head_forward(xs, Wq2, Wk2, Wv2, seq_len, embd, head_dim)
    var attn_out = zeros(seq_len * attn_dim)
    for t in range(seq_len):
        var out_offset = t * attn_dim
        var c1_offset = t * head_dim
        var c2_offset = t * head_dim
        for d in range(head_dim):
            attn_out[out_offset + d] = c1.attn_out[c1_offset + d]
            attn_out[out_offset + head_dim + d] = c2.attn_out[c2_offset + d]
    var last_offset = (seq_len - 1) * attn_dim
    var last_at = attn_out + last_offset
    var h_pre = linear(last_at, W1, hidden, attn_dim)
    var h = relu(h_pre, hidden)
    var logits = linear(h, Wout, vocab, hidden)
    return logits

def generate_text[origin_Wte: Origin, origin_Wout: Origin, origin_W1: Origin, origin_Wq1: Origin, origin_Wk1: Origin, origin_Wv1: Origin, origin_Wq2: Origin, origin_Wk2: Origin, origin_Wv2: Origin](
    Wte: UnsafePointer[Scalar[F64], origin_Wte],
    Wout: UnsafePointer[Scalar[F64], origin_Wout],
    W1: UnsafePointer[Scalar[F64], origin_W1],
    Wq1: UnsafePointer[Scalar[F64], origin_Wq1],
    Wk1: UnsafePointer[Scalar[F64], origin_Wk1],
    Wv1: UnsafePointer[Scalar[F64], origin_Wv1],
    Wq2: UnsafePointer[Scalar[F64], origin_Wq2],
    Wk2: UnsafePointer[Scalar[F64], origin_Wk2],
    Wv2: UnsafePointer[Scalar[F64], origin_Wv2],
    id_to_char: List[String],
    BOS: Int,
    max_new_tokens: Int = 20,
    temperature: Float64 = 0.8) -> String:
    var tokens = zeros(max_new_tokens + 1)
    tokens[0] = Scalar[F64](BOS)
    var current_len = 1

    for _ in range(max_new_tokens):
        # Forward
        var logits = model_forward(tokens, current_len, Wte, Wout, W1, Wq1, Wk1, Wv1, Wq2, Wk2, Wv2, len(id_to_char), 16, 32, 8)
        var next_id = sample_from_logits(logits, len(id_to_char), temperature)
        tokens[current_len] = Scalar[F64](next_id)
        current_len += 1

    # Convertir tokens a texto
    var result = String()
    for i in range(1, current_len):
        var tid = Int(tokens[i])
        if tid < len(id_to_char) - 1:
            result += id_to_char[tid]
    return result

def main() raises:
    seed(42)

    if not Path("input.txt").exists():
        var urllib = Python.import_module("urllib.request")
        urllib.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
            "input.txt"
        )

    var text = Path("input.txt").read_text()
    var lines = text.split("\n")
    var docs = List[String]()
    for l in lines:
        var s = String(l).strip()
        if s.byte_length() > 0:
            docs.append(String(s))

    print("num docs:", len(docs))
    var chars = List[String]()
    for d in docs:
        for c in d.codepoints():
            var ch = String(c)
            var found = False
            for existing in chars:
                if existing == ch:
                    found = True
                    break
            if not found:
                chars.append(ch)

    for i in range(len(chars)):
        for j in range(len(chars) - i - 1):
            if chars[j] > chars[j + 1]:
                var tmp = chars[j]
                chars[j] = chars[j + 1]
                chars[j + 1] = tmp

    var char_to_id = Dict[String, Int]()
    var id_to_char = List[String]()

    for i in range(len(chars)):
        char_to_id[chars[i]] = i
        id_to_char.append(chars[i])

    id_to_char.append("<BOS>")
    var BOS = len(chars)
    var vocab = len(chars) + 1
    print("Vocab size:", vocab)
    print("BOS token index:", BOS)
    # Hiperparámetros
    var embd = 16
    var head_dim = 8
    var hidden = 32
    var lr = 1e-3
    var beta1 = 0.9
    var beta2 = 0.999
    var eps = 1e-8
    var scale_embd = 1.0 / (Float64(embd) ** 0.5)
    var scale_hidden = 1.0 / (Float64(hidden) ** 0.5)
    # Inicialización de pesos
    var Wte  = rand_matrix_scaled(vocab, embd, scale_embd)
    var Wout = rand_matrix_scaled(vocab, hidden, scale_hidden)
    var W1   = rand_matrix_scaled(hidden, 2 * head_dim, scale_embd)
    var Wq1  = rand_matrix_scaled(head_dim, embd, scale_embd)
    var Wk1  = rand_matrix_scaled(head_dim, embd, scale_embd)
    var Wv1  = rand_matrix_scaled(head_dim, embd, scale_embd)
    var Wq2  = rand_matrix_scaled(head_dim, embd, scale_embd)
    var Wk2  = rand_matrix_scaled(head_dim, embd, scale_embd)
    var Wv2  = rand_matrix_scaled(head_dim, embd, scale_embd)
    # Adam
    var mWte = zeros(vocab * embd)
    var vWte = zeros(vocab * embd)
    var mWout = zeros(vocab * hidden)
    var vWout = zeros(vocab * hidden)
    var mW1 = zeros(hidden * 2 * head_dim)
    var vW1 = zeros(hidden * 2 * head_dim)
    var mWq1 = zeros(head_dim * embd); var vWq1 = zeros(head_dim * embd)
    var mWk1 = zeros(head_dim * embd); var vWk1 = zeros(head_dim * embd)
    var mWv1 = zeros(head_dim * embd); var vWv1 = zeros(head_dim * embd)
    var mWq2 = zeros(head_dim * embd); var vWq2 = zeros(head_dim * embd)
    var mWk2 = zeros(head_dim * embd); var vWk2 = zeros(head_dim * embd)
    var mWv2 = zeros(head_dim * embd); var vWv2 = zeros(head_dim * embd)
    print("Mostrando en formato: | Step | Perdida | Nombre |\n")
    var max_steps = 8000
    var context_len = 10

    for step in range(max_steps):
        var t = step + 1
        var idx_name = Int(random_float64() * Float64(len(docs)))
        var name = docs[idx_name]
        var name_tokens = zeros(context_len + 1)
        name_tokens[0] = Scalar[F64](BOS)
        var token_count = 1

        for c in name.codepoints():
            if token_count >= context_len + 1:
                break
            var ch = String(c)
            var token_id: Int = BOS
            for i in range(len(chars)):
                if chars[i] == ch:
                    token_id = i
                    break
            name_tokens[token_count] = Scalar[F64](token_id)
            token_count += 1

        var seq_len_used = token_count - 1
        var tokens_batch = zeros(seq_len_used)
        var targets_batch = zeros(seq_len_used)

        for i in range(seq_len_used):
            tokens_batch[i] = name_tokens[i]
            targets_batch[i] = name_tokens[i + 1]
        var r = train_step(tokens_batch, targets_batch, seq_len_used, Wte, Wout, W1, Wq1, Wk1, Wv1, Wq2, Wk2, Wv2, lr, vocab, embd, hidden, head_dim)
        var rWte  = adam_update(Wte, r.g_Wte, mWte, vWte, vocab * embd, t, lr, beta1, beta2, eps)
        var rWout = adam_update(Wout, r.g_Wout, mWout, vWout, vocab * hidden, t, lr, beta1, beta2, eps)
        var rW1   = adam_update(W1, r.g_W1, mW1, vW1, hidden * 2 * head_dim, t, lr, beta1, beta2, eps)
        var rWq1  = adam_update(Wq1, r.g_Wq1, mWq1, vWq1, head_dim * embd, t, lr, beta1, beta2, eps)
        var rWk1  = adam_update(Wk1, r.g_Wk1, mWk1, vWk1, head_dim * embd, t, lr, beta1, beta2, eps)
        var rWv1  = adam_update(Wv1, r.g_Wv1, mWv1, vWv1, head_dim * embd, t, lr, beta1, beta2, eps)
        var rWq2  = adam_update(Wq2, r.g_Wq2, mWq2, vWq2, head_dim * embd, t, lr, beta1, beta2, eps)
        var rWk2  = adam_update(Wk2, r.g_Wk2, mWk2, vWk2, head_dim * embd, t, lr, beta1, beta2, eps)
        var rWv2  = adam_update(Wv2, r.g_Wv2, mWv2, vWv2, head_dim * embd, t, lr, beta1, beta2, eps)

        Wte = rWte.w; mWte = rWte.m; vWte = rWte.v
        Wout = rWout.w; mWout = rWout.m; vWout = rWout.v
        # Head de atencion 1
        W1 = rW1.w; mW1 = rW1.m; vW1 = rW1.v
        Wq1 = rWq1.w; mWq1 = rWq1.m; vWq1 = rWq1.v
        Wk1 = rWk1.w; mWk1 = rWk1.m; vWk1 = rWk1.v
        Wv1 = rWv1.w; mWv1 = rWv1.m; vWv1 = rWv1.v
        # Head de atencion 2
        Wq2 = rWq2.w; mWq2 = rWq2.m; vWq2 = rWq2.v
        Wk2 = rWk2.w; mWk2 = rWk2.m; vWk2 = rWk2.v
        Wv2 = rWv2.w; mWv2 = rWv2.m; vWv2 = rWv2.v

        if step % 200 == 0 or step < 10:
            print("step:", step, "loss:", r.loss, "nombre:", name)
