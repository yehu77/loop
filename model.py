"""Single-farm conditional DiT: one token per hour, recurrence within denoising."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "loop"  # base: one core; loop: reused core; untied: distinct cores
    horizon: int = 24
    nwp_features: int = 4
    hidden_dim: int = 64
    heads: int = 4
    ffn_dim: int = 256
    blocks_per_core: int = 2
    rounds: int = 2
    attention_scope: str = "full"
    history_group: str | None = None
    history_projection_seed: int = 20042

    def __post_init__(self):
        if self.variant not in ("base", "loop", "untied"):
            raise ValueError("variant must be base, loop, or untied")
        if self.attention_scope not in ("full", "diagonal"):
            raise ValueError("attention_scope must be full or diagonal")
        if self.history_group is not None and (self.history_group not in ("H0", "H1", "H24")
                or self.attention_scope != "full" or self.horizon != 24
                or (self.variant != "base" and self.rounds != 2)):
            raise ValueError("History requires H0/H1/H24, full attention, H=24 and fixed R2 for LOOP/UNTIED")
        dimensions = (self.horizon, self.nwp_features, self.hidden_dim, self.heads,
                      self.ffn_dim, self.blocks_per_core, self.rounds)
        if any(not isinstance(value, int) or value < 1 for value in dimensions):
            raise ValueError("All dimensions and rounds must be positive integers")
        if self.hidden_dim % 2 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be even and divisible by heads")


def sinusoidal_embedding(positions, dim):
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(dim // 2, device=positions.device).float()
        / (dim // 2)
    )
    phase = positions.float().unsqueeze(-1) * frequencies
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, attention_scope="full"):
        super().__init__()
        self.heads = heads
        self.attention_scope = attention_scope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x, key_visibility=None):
        batch, horizon, dim = x.shape
        qkv = self.qkv(x).reshape(batch, horizon, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if key_visibility is not None:
            if (self.attention_scope != "full" or key_visibility.dtype != torch.bool
                    or key_visibility.shape != (batch, horizon)
                    or not key_visibility.any(dim=-1).all()):
                raise ValueError("Invalid boolean key visibility mask")
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=key_visibility[:, None, None, :], dropout_p=0.0, is_causal=False)
        elif self.attention_scope == "diagonal":
            # SDPA boolean masks use True for allowed keys. Keep Q/K/V tensors;
            # with one allowed key, Q/K cannot affect the attention weights.
            mask = torch.eye(horizon, dtype=torch.bool, device=x.device)
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        else:
            # Preserve the historical full-attention call exactly.
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.projection(attended.transpose(1, 2).reshape(batch, horizon, dim))


class AdaLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.attention = SelfAttention(dim, config.heads, config.attention_scope)
        self.ffn = nn.Sequential(nn.Linear(dim, config.ffn_dim), nn.GELU(),
                                 nn.Linear(config.ffn_dim, dim))
        self.modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, h, time_embedding, key_visibility=None):
        s_a, b_a, g_a, s_f, b_f, g_f = self.modulator(time_embedding).unsqueeze(1).chunk(6, -1)
        z = (1 + s_a) * self.norm_attention(h) + b_a
        attended = self.attention(z) if key_visibility is None else self.attention(z, key_visibility)
        h = h + g_a * attended
        z = (1 + s_f) * self.norm_ffn(h) + b_f
        return h + g_f * self.ffn(z)


class Core(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adapter = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(AdaLNBlock(config) for _ in range(config.blocks_per_core))

    def forward(self, h, evidence, time_embedding, key_visibility=None):
        h = self.adapter(torch.cat((h, evidence), dim=-1))
        for block in self.blocks:
            h = block(h, time_embedding) if key_visibility is None else block(h, time_embedding, key_visibility)
        return h


class OutputHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.projection = nn.Linear(dim, 1)

    def forward(self, h, time_embedding):
        scale, shift = self.modulator(time_embedding).unsqueeze(1).chunk(2, -1)
        return self.projection((1 + scale) * self.norm(h) + shift)


class ConditionalLoopDiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        config = self.config
        dim = config.hidden_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(config.nwp_features + 4, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.input_projection = nn.Linear(1 + dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        core_count = config.rounds if config.variant == "untied" else 1
        self.cores = nn.ModuleList(Core(config) for _ in range(core_count))
        self.output_head = OutputHead(dim)
        self.register_buffer("position_encoding", sinusoidal_embedding(
            torch.arange(config.horizon), dim
        ).unsqueeze(0))
        self.initialize_weights()
        if config.history_group is not None:
            # Add history only after canonical BASE initialization. Isolate its
            # RNG use so H0/H1/H24 and the legacy common tensors start identically.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.history_projection_seed)
                self.history_projection = nn.Linear(1, dim)
                nn.init.xavier_uniform_(self.history_projection.weight)
                nn.init.zeros_(self.history_projection.bias)
            self.position_encoding.copy_(sinusoidal_embedding(torch.arange(1, 25), dim).unsqueeze(0))
            self.register_buffer("history_position_encoding", sinusoidal_embedding(
                torch.arange(-24, 0), dim).unsqueeze(0))

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            for core in self.cores:
                core.adapter.weight.zero_()
                core.adapter.weight[:, :self.config.hidden_dim].copy_(
                    torch.eye(self.config.hidden_dim, device=core.adapter.weight.device)
                )
            for module in self.modules():
                if isinstance(module, (AdaLNBlock, OutputHead)):
                    nn.init.zeros_(module.modulator[-1].weight)
                    nn.init.zeros_(module.modulator[-1].bias)
            nn.init.zeros_(self.output_head.projection.weight)
            nn.init.zeros_(self.output_head.projection.bias)

    def encode_condition(self, nwp, calendar, history_power=None):
        if nwp.ndim != 3 or nwp.shape[1:] != (self.config.horizon, self.config.nwp_features):
            raise ValueError("nwp must have shape [B, horizon, nwp_features]")
        if calendar.shape != (*nwp.shape[:2], 4):
            raise ValueError("calendar must have shape [B, horizon, 4]")
        future = self.condition_mlp(torch.cat((nwp, calendar), dim=-1))
        if self.config.history_group is None:
            if history_power is not None:
                raise ValueError("Legacy model cannot accept history")
            return future
        if history_power is None or history_power.shape != (*nwp.shape[:2], 1):
            raise ValueError("History model requires raw capacity power [B, 24, 1]")
        from history_data import visible_history
        # This is the only history scale transform, before the shared projection.
        values, visible = visible_history(history_power, self.config.history_group)
        if not torch.isfinite(values).all() or ((values < -1) | (values > 1)).any():
            raise ValueError("Visible history must contain finite capacity fractions")
        history = self.history_projection(values) + self.history_position_encoding
        return {"future": future, "history": history,
                "key_visibility": torch.cat((visible, torch.ones_like(visible)), dim=1)}

    def denoise_with_cached_condition(self, x_t, condition, t, rounds=None, return_stages=False):
        if x_t.ndim != 3 or x_t.shape[1:] != (self.config.horizon, 1):
            raise ValueError("x_t must have shape [B, horizon, 1]")
        history_enabled = self.config.history_group is not None
        future = condition["future"] if history_enabled else condition
        if future.shape != (*x_t.shape[:2], self.config.hidden_dim):
            raise ValueError("encoded condition has an incompatible shape")
        if t.shape != (x_t.shape[0],):
            raise ValueError("t must have shape [B]")
        default_rounds = 1 if self.config.variant == "base" else self.config.rounds
        rounds = default_rounds if rounds is None else rounds
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if self.config.variant == "untied" and rounds != len(self.cores):
            raise ValueError("untied model must execute all its independent cores")
        if history_enabled and return_stages:
            raise ValueError("History interface supports final-only readout")
        if history_enabled and rounds != default_rounds:
            raise ValueError("History interface uses the fixed architecture execution count")
        evidence = self.input_projection(torch.cat((x_t, future), dim=-1))
        evidence = evidence + self.position_encoding
        if history_enabled:
            if condition["history"].shape != future.shape:
                raise ValueError("Cached history has an incompatible shape")
            evidence = torch.cat((condition["history"], evidence), dim=1)
        time_embedding = self.time_mlp(sinusoidal_embedding(t, self.config.hidden_dim))
        h = evidence
        outputs = [] if return_stages else None
        for r in range(rounds):
            core = self.cores[r] if self.config.variant == "untied" else self.cores[0]
            h = (core(h, evidence, time_embedding, condition["key_visibility"])
                 if history_enabled else core(h, evidence, time_embedding))
            if return_stages:
                outputs.append(self.output_head(h, time_embedding))
        if return_stages:
            return tuple(outputs)
        return self.output_head(h[:, 24:] if history_enabled else h, time_embedding)

    def forward_stages(self, x_t, nwp, calendar, t):
        """Two shared-head readouts from one uninterrupted two-Core pass."""
        if self.config.history_group is not None:
            raise ValueError("History interface supports final-only readout")
        if self.config.variant not in ("loop", "untied") or self.config.rounds != 2:
            raise ValueError("Intermediate supervision requires two Core executions")
        condition = self.encode_condition(nwp, calendar)
        return self.denoise_with_cached_condition(x_t, condition, t, return_stages=True)

    def forward(self, x_t, nwp, calendar, t, rounds=None, history_power=None):
        condition = self.encode_condition(nwp, calendar, history_power)
        return self.denoise_with_cached_condition(x_t, condition, t, rounds)
