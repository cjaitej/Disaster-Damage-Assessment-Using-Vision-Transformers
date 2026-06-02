import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


def make_group_norm(channels: int):
    """Return a valid GroupNorm for any channel count."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)



class OCA(nn.Module):
    """
    Overlapping Cross-Attention module.

    Computes local attention using sliding windows with overlap.
    Improves feature continuity across window boundaries while
    keeping computation efficient.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        overlap: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.overlap = overlap
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ext = window_size + 2 * overlap
        table_size = (2 * ext - 1) * (2 * ext - 1)
        self.rel_pos_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        self._build_rel_pos_index(window_size, ext)

    def _build_rel_pos_index(self, ws: int, ext: int):
        coords_q_h = torch.arange(ws) + self.overlap
        coords_q_w = torch.arange(ws) + self.overlap
        coords_k_h = torch.arange(ext)
        coords_k_w = torch.arange(ext)

        gq = torch.stack(torch.meshgrid(coords_q_h, coords_q_w, indexing="ij"))
        gk = torch.stack(torch.meshgrid(coords_k_h, coords_k_w, indexing="ij"))

        gq_flat = gq.flatten(1)
        gk_flat = gk.flatten(1)

        rel = gq_flat[:, :, None] - gk_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()

        rel[:, :, 0] += ext - 1
        rel[:, :, 1] += ext - 1
        rel[:, :, 0] *= 2 * ext - 1
        index = rel.sum(-1)
        self.register_buffer("rel_pos_index", index)

    def _tile(self, x: torch.Tensor, ws: int, overlap: int):
        B, Hp_ext, Wp_ext, C = x.shape
        ext = ws + 2 * overlap
        nH = (Hp_ext - 2 * overlap) // ws
        nW = (Wp_ext - 2 * overlap) // ws

        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        tiles = F.unfold(x_nchw, kernel_size=ext, stride=ws)
        tiles = tiles.permute(0, 2, 1).reshape(B * nH * nW, ext * ext, C)
        return tiles, nH, nW

    def _tile_center(self, x: torch.Tensor, ws: int):
        B, Hp, Wp, C = x.shape
        nH, nW = Hp // ws, Wp // ws
        x = x.view(B, nH, ws, nW, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.reshape(B * nH * nW, ws * ws, C), nH, nW

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        ws = self.window_size
        ovlp = self.overlap

        x2d = x.view(B, H, W, C)

        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        x2d = F.pad(x2d, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x2d.shape

        x2d_ext = F.pad(
            x2d.permute(0, 3, 1, 2),
            (ovlp, ovlp, ovlp, ovlp),
            mode="reflect",
        ).permute(0, 2, 3, 1)

        ext_tiles, nH, nW = self._tile(x2d_ext, ws, ovlp)
        ctr_tiles, _, _ = self._tile_center(x2d, ws)

        N_tiles = B * nH * nW
        Nq = ws * ws
        Nkv = (ws + 2 * ovlp) ** 2

        q = self.q(ctr_tiles).reshape(N_tiles, Nq, self.num_heads, C // self.num_heads)
        kv = self.kv(ext_tiles).reshape(N_tiles, Nkv, 2, self.num_heads, C // self.num_heads)

        q = q.permute(0, 2, 1, 3)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        bias = bias.view(Nq, Nkv, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(N_tiles, Nq, C)
        out = self.proj_drop(self.proj(out))

        out = out.view(B, nH, nW, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, Hp, Wp, C)

        out = out[:, :H, :W, :].contiguous().view(B, H * W, C)
        return out



class ChannelAttentionFusion(nn.Module):
    """
    Channel attention module.

    Applies global average pooling followed by a small MLP to
    generate channel-wise weights, emphasizing important features.
    """
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        gap = x.mean(dim=1)
        weights = self.fc(gap).unsqueeze(1)
        return x * weights



class GlobalSparseAttention(nn.Module):
    """
    Global sparse attention module.

    Captures long-range dependencies by attending to a downsampled
    set of tokens instead of the full feature map.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride: int = 2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.stride = stride
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, N, C = x.shape
        s = self.stride

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv_tokens = x.view(B, H, W, C)[:, ::s, ::s, :].reshape(B, -1, C)
        kv_tokens = kv_tokens + self.pos_embed
        kv = self.kv(kv_tokens).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))



class LearnableGateFusion(nn.Module):
    """
    Feature fusion using a learnable gate.

    Combines local and global features by learning a balance between them.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim // 2, bias=False),
            nn.GELU(),
            nn.Linear(dim // 2, dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor):
        alpha = self.gate(torch.cat([local_feat, global_feat], dim=-1))
        return alpha * local_feat + (1.0 - alpha) * global_feat



class HATBlock(nn.Module):
    """
    Hybrid Attention Transformer block.

    Combines local attention, global attention, and feed-forward layers
    with residual connections.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        overlap: int = 3,
        global_stride: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        se_reduction: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.local_attn = OCA(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            overlap=overlap,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.channel_attn = ChannelAttentionFusion(dim=dim, reduction=se_reduction)

        self.global_attn = GlobalSparseAttention(
            dim=dim,
            num_heads=num_heads,
            stride=global_stride,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.gate_fusion = LearnableGateFusion(dim=dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, H: int, W: int):
        shortcut = x
        x_norm = self.norm1(x)

        local_out = self.local_attn(x_norm, H, W)
        local_out = self.channel_attn(local_out)

        global_out = self.global_attn(x_norm, H, W)

        fused = self.gate_fusion(local_out, global_out)
        x = shortcut + self.drop_path(fused)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x



class PatchEmbed(nn.Module):
    """
    Patch embedding module.

    Converts input images into embedded feature representations.
    """
    def __init__(self, img_size=512, patch_size=4, in_chans=3, embed_dim=80):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            make_group_norm(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            make_group_norm(embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H_out, W_out



class PatchMerge(nn.Module):
    """
    Patch merging layer.

    Reduces spatial resolution while increasing channel dimensions.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        pad_b = H % 2
        pad_r = W % 2
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = H + pad_b, W + pad_r

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)

        H_out = H_pad // 2
        W_out = W_pad // 2
        return self.reduction(self.norm(x)), H_out, W_out



class HATEncoder(nn.Module):
    """
    Hierarchical encoder using HAT blocks.

    Produces multi-scale feature maps from the input image.
    """
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 80,
        depths: tuple = (2, 2, 4, 2),
        num_heads: tuple = (2, 4, 8, 16),
        window_size: int = 8,
        overlap: int = 3,
        global_stride: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        total_blocks = sum(depths)
        dpr = [r.item() for r in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        block_idx = 0
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage = nn.ModuleList([
                HATBlock(
                    dim=dims[i],
                    num_heads=heads,
                    window_size=window_size,
                    overlap=overlap,
                    global_stride=global_stride,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[block_idx + j],
                )
                for j in range(depth)
            ])
            block_idx += depth
            self.stages.append(stage)
            if i < len(depths) - 1:
                self.merges.append(PatchMerge(dims[i]))

    def forward(self, x: torch.Tensor):
        x, H, W = self.patch_embed(x)
        features = []

        for i, stage in enumerate(self.stages):
            for block in stage:
                x = block(x, H, W)
            B, _, C = x.shape
            features.append(x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous())
            if i < len(self.stages) - 1:
                x, H, W = self.merges[i](x, H, W)

        return features   # [F1, F2, F3, F4]



class HATEncoderWrapper(nn.Module):
    """
    Wrapper for encoder initialization and configuration.
    """
    def __init__(self):
        super().__init__()
        self.encoder = HATEncoder(
            embed_dim=80,
            num_heads=(2, 4, 8, 16),
            depths=(2, 2, 4, 2),
            drop_path_rate=0.3,
        )
        self.out_channels = [80, 160, 320, 640]

    def forward(self, x):
        return tuple(self.encoder(x))



class FusionAttention(nn.Module):
    """
    Feature fusion module.

    Combines multi-scale features using top-down pathway and attention.
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()

        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False)
            for c in in_channels_list
        ])

        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                make_group_norm(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(len(in_channels_list) - 1)
        ])

        hidden = max(out_channels // 4, 16)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        laterals = [l(f) for l, f in zip(self.laterals, features)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = self.td_convs[i - 1](laterals[i - 1] + upsampled)

        out = laterals[0]
        return out * self.attn(out)



class SkipAttentionGate(nn.Module):
    """
    Skip connection attention gate.

    Filters encoder features before merging with decoder.
    """
    def __init__(self, gate_ch, skip_ch):
        super().__init__()
        inter_ch = max(skip_ch // 2, 16)
        self.W_g = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.W_s = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False)
        self.norm_g = make_group_norm(inter_ch)
        self.norm_s = make_group_norm(inter_ch)

    def forward(self, g, s):
        alpha = torch.sigmoid(
            self.psi(
                F.relu(self.norm_g(self.W_g(g)) + self.norm_s(self.W_s(s)), inplace=True)
            )
        )
        return s * alpha



class DecoderBlock(nn.Module):
    """
    Decoder block.

    Upsamples features and fuses them with skip connections.
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.skip_norm = make_group_norm(skip_ch)
        self.attn_gate = SkipAttentionGate(gate_ch=in_ch, skip_ch=skip_ch)

        fused_ch = in_ch + skip_ch

        self.proj = (
            nn.Conv2d(fused_ch, out_ch, kernel_size=1, bias=False)
            if fused_ch != out_ch else nn.Identity()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(fused_ch, out_ch, 3, padding=1, bias=False),
            make_group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]

        skip = self.skip_norm(skip)
        skip = self.attn_gate(x, skip)

        fused = torch.cat([x, skip], dim=1)
        return self.conv(fused) + self.proj(fused)



class UNetDecoder(nn.Module):
    """
    U-Net style decoder.

    Reconstructs spatial resolution progressively.
    """
    def __init__(self, encoder_channels, base_ch=160):
        super().__init__()

        enc_ch = encoder_channels[::-1]
        n = len(enc_ch) - 1

        self.blocks = nn.ModuleList()
        self.out_ch = base_ch

        prev_ch = enc_ch[0]
        for i in range(n):
            self.blocks.append(DecoderBlock(prev_ch, enc_ch[i + 1], base_ch))
            prev_ch = base_ch

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        for i, block in enumerate(self.blocks):
            x = block(x, features[i + 1])
        return x



class BoundaryRefinementAttention(nn.Module):
    """
    Boundary refinement module.

    Enhances edges and refines feature maps for better segmentation.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()

        self.edge = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            make_group_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            make_group_norm(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape

        edge_map = self.edge(x)
        spatial_gate = torch.sigmoid(edge_map)

        ch_gate = self.channel_attn(x).view(B, C, 1, 1)
        x_attended = x * spatial_gate * ch_gate

        refined = self.relu(self.refine(x_attended) + x_attended)
        return x + refined, edge_map



class SegmentationHead(nn.Module):
    """
    Segmentation head.

    Produces final class predictions from features.
    """
    def __init__(self, in_channels, num_classes, inter_channels=None, dropout=0.1):
        super().__init__()

        inter_channels = inter_channels or max(in_channels // 2, 64)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Conv2d(inter_channels, num_classes, kernel_size=1)

    def forward(self, x, output_size=None):
        x = self.conv(x)
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.classifier(x)



class FullModelHAT(nn.Module):
    """
    Complete segmentation model.

    Integrates encoder, decoder, and refinement modules.
    """
    def __init__(self, num_classes=11, fusion_out_channels=256, decoder_base_ch=160):
        super().__init__()

        self.encoder = HATEncoderWrapper()
        enc_channels = self.encoder.out_channels  # [80, 160, 320, 640]

        self.fusion = FusionAttention(enc_channels, fusion_out_channels)
        self.decoder = UNetDecoder(enc_channels[:-1] + [fusion_out_channels], base_ch=decoder_base_ch)
        self.refine = BoundaryRefinementAttention(self.decoder.out_ch)
        self.head = SegmentationHead(self.decoder.out_ch, num_classes)

        self.aux_head = SegmentationHead(self.decoder.out_ch, num_classes)
        self.aux_block_idx = len(self.decoder.blocks) // 2

    def forward(self, x):
        features = self.encoder(x)  # [F1, F2, F3, F4]

        fused_context = self.fusion(features)  # F1-resolution context
        fused_context = F.interpolate(
            fused_context,
            size=features[-1].shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Decoder input order must be [F1, F2, F3, bottleneck]
        dec_features = list(features[:-1]) + [fused_context]

        # Manual decode to capture midpoint feature for deep supervision
        feats = dec_features[::-1]
        decoded = feats[0]
        aux_feat = None

        for i, block in enumerate(self.decoder.blocks):
            decoded = block(decoded, feats[i + 1])
            if i == self.aux_block_idx:
                aux_feat = decoded

        refined, edge_map = self.refine(decoded)
        out = self.head(refined, output_size=x.shape[2:])

        if self.training:
            aux_out = None
            if aux_feat is not None:
                aux_out = self.aux_head(aux_feat, output_size=x.shape[2:])
            return out, aux_out, edge_map

        return out, edge_map



def count_parameters(model):
    """
    Utility function to compute total and trainable parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}   ({total/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable:,}   ({trainable/1e6:.2f}M)")
    return total, trainable


if __name__ == "__main__":
    model = FullModelHAT(num_classes=11, fusion_out_channels=256, decoder_base_ch=160)
    count_parameters(model)

    x = torch.randn(1, 3, 384, 384)

    # Train mode: returns main output, auxiliary output, edge map
    model.train()
    with torch.no_grad():
        seg, aux, edge = model(x)
    print(f"[Train mode]")
    print(f"Input  : {tuple(x.shape)}")
    print(f"Seg out: {tuple(seg.shape)}   (expected [1, 11, 384, 384])")
    print(f"Aux out: {tuple(aux.shape) if aux is not None else None}")
    print(f"Edge   : {tuple(edge.shape)}")

    # Eval mode: returns main output + edge map
    model.eval()
    with torch.no_grad():
        seg, edge = model(x)
    print(f"[Eval mode]")
    print(f"Seg out: {tuple(seg.shape)}   (expected [1, 11, 384, 3d84])")
    print(f"Edge   : {tuple(edge.shape)}")
