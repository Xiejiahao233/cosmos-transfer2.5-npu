# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import SACConfig
from cosmos_transfer2._src.predict2.networks.a2a_cp import LocalWindowA2AAttnOp
from cosmos_transfer2._src.transfer2.networks.minimal_v4_lvg_dit_control_vace import MinimalV4LVGControlVaceDiT

TRANSFER2_CONTROL2WORLD_NET_2B = L(MinimalV4LVGControlVaceDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    extra_per_block_abs_pos_emb=False,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=1.0,
    sac_config=SACConfig(),
)

TRANSFER2_CONTROL2WORLD_NET_14B = copy.deepcopy(TRANSFER2_CONTROL2WORLD_NET_2B)
TRANSFER2_CONTROL2WORLD_NET_14B.model_channels = 5120
TRANSFER2_CONTROL2WORLD_NET_14B.num_heads = 40
TRANSFER2_CONTROL2WORLD_NET_14B.num_blocks = 36


def _enable_local_window_self_attention(net_cfg, chunk_size: int = 4096, min_seq_len: int = 8192):
    """Replace dense self-attention with a local-window op for long video sequences.

    The profiling bottleneck was FlashAttentionScore on Q/K/V with sequence length
    84480. LocalWindowA2AAttnOp only changes matching long self-attention and
    keeps cross-attention/global short attention untouched. Runtime override:
    COSMOS_LOCAL_ATTN_DISABLE=1 reverts to the original dense path.
    """
    net_cfg.n_dense_blocks = 0
    net_cfg.gna_parameters = {
        "window_size": (1, 1, 1),
        "local_attn_op_cls": LocalWindowA2AAttnOp,
        "chunk_size": chunk_size,
        "min_seq_len": min_seq_len,
    }


_enable_local_window_self_attention(TRANSFER2_CONTROL2WORLD_NET_2B)
_enable_local_window_self_attention(TRANSFER2_CONTROL2WORLD_NET_14B)


def register_net():
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="transfer2_control2world_net_2B",
        node=TRANSFER2_CONTROL2WORLD_NET_2B,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="transfer2_control2world_net_14B",
        node=TRANSFER2_CONTROL2WORLD_NET_14B,
    )
