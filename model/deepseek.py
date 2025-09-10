import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

# DeepSeek's custom kernel functions for quantization and efficient GEMM
# These are implemented in DeepSeek's kernel.py for optimal performance
from kernel import act_quant, weight_dequant, fp8_gemm

# Global configuration matching DeepSeek's distributed setup
world_size = 1
rank = 0
block_size = 128  # DeepSeek's block size for quantization
gemm_impl: Literal["bf16", "fp8"] = "bf16"  # DeepSeek's GEMM implementation choice
attn_impl: Literal["naive", "absorb"] = "absorb"  # DeepSeek's attention optimization

@dataclass
class ModelArgs:
    """
    DeepSeek V3 model configuration parameters.
    These values are based on DeepSeek's architecture choices and hyperparameters.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4  # Extended context length with YARN
    dtype: Literal["bf16", "fp8"] = "bf16"  # DeepSeek's precision choices
    vocab_size: int = 102400  # DeepSeek's vocabulary size
    dim: int = 2048  # Model dimension
    inter_dim: int = 10944  # Intermediate dimension for MLP
    moe_inter_dim: int = 1408  # Intermediate dimension for MoE experts
    n_layers: int = 27  # Total transformer layers
    n_dense_layers: int = 1  # Layers using dense MLP before switching to MoE
    n_heads: int = 16  # Number of attention heads

    # MoE configuration - DeepSeek's expert routing strategy
    n_routed_experts: int = 64  # Total experts in MoE layer
    n_shared_experts: int = 2  # Always-active shared experts
    n_activated_experts: int = 6  # Experts activated per token
    n_expert_groups: int = 1  # Expert grouping for efficient routing
    n_limited_groups: int = 1  # Limited groups for top-k selection
    score_func: Literal["softmax", "sigmoid"] = "softmax"  # DeepSeek's gating function
    route_scale: float = 1.  # Scaling factor for expert weights

    # Multi-head Latent Attention - DeepSeek's efficient attention design
    q_lora_rank: int = 0  # LoRA rank for queries (0 means no LoRA)
    kv_lora_rank: int = 512  # LoRA rank for keys/values compression
    qk_nope_head_dim: int = 128  # Dimension without positional encoding
    qk_rope_head_dim: int = 64  # Dimension with rotary positional encoding
    v_head_dim: int = 128  # Value dimension

    # Rotary Positional Encoding with YARN extension - DeepSeek's long context solution
    original_seq_len: int = 4096  # Original training sequence length
    rope_theta: float = 10000.0  # Base frequency for RoPE
    rope_factor: float = 40  # Scaling factor for extended context
    beta_fast: int = 32  # Fast beta for YARN correction
    beta_slow: int = 1  # Slow beta for YARN correction
    mscale: float = 1.  # Attention scale multiplier for long context

    # Regularization - DeepSeek's dropout configuration
    dropout: float = 0.1


class ParallelEmbedding(nn.Module):
    """
    DeepSeek's distributed embedding layer that splits vocabulary across GPUs.
    Each process handles a portion of the vocabulary for memory efficiency.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DeepSeek's parallel embedding forward pass with all-reduce"""
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)  # DeepSeek's distributed aggregation
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    DeepSeek's quantization-aware linear transformation.
    Supports both FP8 and BF16 formats using their custom kernel functions.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)  # DeepSeek's dequantization
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)  # DeepSeek's activation quantization
        y = fp8_gemm(x, scale, weight, weight.scale)  # DeepSeek's FP8 GEMM
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    DeepSeek's custom linear layer with support for quantized weights.
    Handles both full-precision and quantized weight representations.
    """
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))

        # DeepSeek's quantization scale handling for low-precision weights
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    DeepSeek's column-parallel linear layer for model parallelism.
    Splits output features across devices for distributed training.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias, self.scale_fmt)


class RowParallelLinear(Linear):
    """
    DeepSeek's row-parallel linear layer for model parallelism.
    Splits input features across devices and reduces results.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None, self.scale_fmt)
        if world_size > 1:
            dist.all_reduce(y)  # DeepSeek's distributed reduction
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    DeepSeek's Root Mean Square Normalization implementation.
    More efficient than LayerNorm while maintaining training stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    DeepSeek's implementation of rotary positional encoding frequencies.
    Includes YARN extensions for long context handling.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """DeepSeek's correction dimension calculation for YARN"""
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """DeepSeek's correction range calculation for YARN"""
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """DeepSeek's linear ramp function for smooth interpolation"""
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # DeepSeek's frequency calculation with YARN extensions
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # DeepSeek's complex exponential
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    DeepSeek's application of rotary positional embeddings.
    Uses complex numbers for efficient rotation operations.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    DeepSeek's Multi-head Latent Attention implementation.
    Key innovation: splits attention heads into positional and non-positional parts,
    uses low-rank compression for KV, and offers efficient KV caching strategies.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # DeepSeek's query projection with optional LoRA
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # DeepSeek's key-value projection with low-rank compression
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5

        # DeepSeek's attention scaling for extended sequences (YARN)
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # DeepSeek's KV cache implementation choices
        if attn_impl == "naive":
            # Traditional full KV cache
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len,
                                                        self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len,
                                                        self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # DeepSeek's memory-efficient "absorb" implementation
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len,
                                                         self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len,
                                                         self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """DeepSeek's MLA forward pass with efficient KV handling"""
        batch_size, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # Process query projections (DeepSeek's optional LoRA path)
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(batch_size, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)  # Apply RoPE to positional part

        # Process key-value projections (DeepSeek's low-rank compression)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        if attn_impl == "naive":
            # Traditional attention implementation
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(batch_size, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)

            # Update KV cache
            self.k_cache[:batch_size, start_pos:end_pos] = k
            self.v_cache[:batch_size, start_pos:end_pos] = v

            # Compute attention scores
            scores = torch.einsum("bshd,bthd->bhst", q, self.k_cache[:batch_size, :end_pos]) * self.softmax_scale
        else:
            # DeepSeek's memory-efficient "absorb" implementation
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(
                self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])

            # Update compressed KV cache (DeepSeek's memory optimization)
            self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:batch_size, start_pos:end_pos] = k_pe.squeeze(2)

            # Compute attention with compressed representation
            scores = (torch.einsum("bshc,btc->bhst", q_nope, self.kv_cache[:batch_size, :end_pos]) +
                      torch.einsum("bshr,btr->bhst", q_pe, self.pe_cache[:batch_size, :end_pos])) * self.softmax_scale

        # Apply attention mask
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax and attention output
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)

        if attn_impl == "naive":
            x = torch.einsum("bhst,bthd->bshd", scores, self.v_cache[:batch_size, :end_pos])
        else:
            x = torch.einsum("bhst,btc->bshc", scores, self.kv_cache[:batch_size, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # Final output projection
        x = self.wo(x.flatten(2))
        return x


class Gate(nn.Module):
    """
    DeepSeek's gating mechanism for Mixture-of-Experts routing.
    Implements both softmax and sigmoid scoring functions with group routing.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DeepSeek's expert routing computation"""
        scores = linear(x, self.weight)

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()

        original_scores = scores

        if self.bias is not None:
            scores = scores + self.bias

        # DeepSeek's group-based routing for efficiency
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)

            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

        # Select top-k experts (DeepSeek's sparse activation)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)

        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)

        weights *= self.route_scale
        return weights.type_as(x), indices


class MLP(nn.Module):
    """
    DeepSeek's Feed-Forward Network with gated linear units.
    Uses SwiGLU activation for better performance.
    """
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DeepSeek's SwiGLU activation: silu(w1(x)) * w3(x)"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    DeepSeek's Mixture-of-Experts implementation.
    Routes tokens to specialized experts with dynamic selection.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([MLP(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DeepSeek's MoE forward pass with expert routing"""
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # Process through selected experts (DeepSeek's sparse computation)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]

        # Process through shared experts (always active)
        z = self.shared_experts(x)

        # Aggregate results across processes
        if world_size > 1:
            dist.all_reduce(y)

        return (y + z).view(shape)


class Block(nn.Module):
    """
    DeepSeek's transformer block structure.
    RMSNorm → Attention → residual → RMSNorm → FFN/MoE → residual
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """DeepSeek's transformer block with residual connections"""
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Complete DeepSeek V3 transformer implementation.
    Embedding → Multiple blocks → RMSNorm → Output projection
    """
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()

        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))

        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None,
                start_pos: int = 0) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """DeepSeek's complete forward pass for training and inference"""
        batch_size, seqlen = tokens.size()
        h = self.embed(tokens)

        if self.training:
            h = self.dropout(h)

        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]

        # Create causal attention mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

        # Process through all transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)

        # Inference mode - return only last token logits
        if not self.training:
            h = h[:, -1]
            logits = self.head(h)

            if world_size > 1:
                all_logits = [torch.empty_like(logits) for _ in range(world_size)]
                dist.all_gather(all_logits, logits)
                logits = torch.cat(all_logits, dim=-1)

            return logits

        # Training mode - return all logits and compute loss
        logits = self.head(h.reshape(-1, h.size(-1)))

        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)

        logits = logits.reshape(batch_size, seqlen, -1)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))

        return logits, loss

    def inference(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """DeepSeek's optimized inference pass"""
        with torch.inference_mode():
            return self.forward(tokens, targets=None, start_pos=start_pos)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)

    # Test inference mode
    logits = model(x)
    print(f"Output size: {logits.size()}")

    # Test training mode
    model.train()
    targets = torch.randint(0, args.vocab_size, (2, 128))
    logits, loss = model(x, targets)
    print(f"Training mode - Logits size: {logits.size()}, Loss: {loss.item()}")
