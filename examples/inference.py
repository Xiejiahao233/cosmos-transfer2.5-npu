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

from pathlib import Path
from typing import Annotated, Union

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.config import (
    BlurConfig,
    DepthConfig,
    EdgeConfig,
    InferenceArguments,
    InferenceOverrides,
    SegConfig,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)
import torch_npu
from torch_npu.contrib import transfer_to_npu

# 导入工具的数据采集接口
#from msprobe.pytorch import PrecisionDebugger, seed_all

# 在模型训练开始前固定随机性
#seed_all()
# 在模型训练开始前实例化PrecisionDebugger
#debugger = PrecisionDebugger(config_path="<msprobe_config.json>")

ControlUnion = Annotated[
    Union[
        Annotated[EdgeConfig, tyro.conf.subcommand("edge")],
        Annotated[DepthConfig, tyro.conf.subcommand("depth")],
        Annotated[BlurConfig, tyro.conf.subcommand("vis")],
        Annotated[SegConfig, tyro.conf.subcommand("seg")],
    ],
    tyro.conf.ConsolidateSubcommandArgs,
]


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path(s) to the inference parameter file(s).
    If multiple files are provided, run "batch" inference. The model will be loaded once and all samples run sequentially.
    If there are different hint keys across the batch, the multicontrol model will be used regardless of the each sample's hint keys.
    """
    setup: SetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: InferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""

    control: ControlUnion = EdgeConfig()
    """Control help. Run control:edge --help for more information about edge etc."""


def main(
    args: Args,
):
    inference_samples, batch_hint_keys = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    if args.setup.benchmark:
        if len(inference_samples) == 1:
            inference_samples = inference_samples * 4
            log.info(f"Repeating inference sample 4 times for benchmarking.")
        # assert len(inference_samples) > 1, "Benchmarking must be run for more than 1 sample."
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    # # 添加Profiling采集扩展配置参数，详细参数介绍可参考下文的参数说明
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     export_type=[
    #         torch_npu.profiler.ExportType.Text
    #     ],
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    #     mstx=False,  # 原参数名msprof_tx改为mstx，新版本依旧兼容原参数名msprof_tx
    #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    #     l2_cache=False,
    #     op_attr=False,
    #     data_simplification=False,
    #     record_op_args=False,
    #     gc_detect_threshold=None,
    #     host_sys=[
    #         torch_npu.profiler.HostSystem.CPU,
    #         torch_npu.profiler.HostSystem.MEM],
    #     sys_io=False,
    #     sys_interconnection=False
    # )
    #
    # # 添加Profiling采集基础配置参数，详细参数介绍可参考下文的参数说明
    # prof = torch_npu.profiler.profile(
    #     activities=[
    #         torch_npu.profiler.ProfilerActivity.CPU,
    #         torch_npu.profiler.ProfilerActivity.NPU
    #     ],
    #     # schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
    #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("<profiling_output_dir>"),
    #     record_shapes=True,
    #     profile_memory=False,
    #     with_stack=True,
    #     with_modules=False,
    #     with_flops=False,
    #     experimental_config=experimental_config)


    from cosmos_transfer2.inference import Control2WorldInference

    inference = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)
    #debugger.start(model=inference.inference_pipeline.model)  # 开启数据dump
    # prof.start()
    inference.generate(inference_samples, output_dir=args.setup.output_dir)
    # prof.step()
    # prof.stop()
    #debugger.stop()  # 关闭数据dump，可继续开启数据dump，采集数据会记录在同一个step中
    #debugger.step()  # 结束数据dump，若继续开启数据dump，采集数据将记录在下一个step中

if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
