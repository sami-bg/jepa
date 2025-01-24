# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import logging
import sys
import wandb

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

PROJECT_NAME = "VJEPA WORLD MODELS"

def main(
    eval_name,
    args_eval,
    resume_preempt=False
):
    logger.info(f'Running evaluation: {eval_name}')
    #  /users/sboughan/scratch/v-jepa-distractors/checkpoints/vjepa/pretrain/ssv2-vjepa-vit_base-384-num_frames16-sampling_rate4-e300/logs
    # returns ssv2-vjepa-vit_base-384-num_frames16-sampling_rate4-e300
    run_name = 'eval_' + f'{eval_name}_' + args_eval['pretrain']['folder'].split('/')[-2]
    wandb.init(project=PROJECT_NAME, name=run_name, config=args_eval, resume="allow", entity='samibg')
    res = importlib.import_module(f'evals.{eval_name}.eval').main(
        args_eval=args_eval,
        resume_preempt=resume_preempt)
    wandb.finish()
    return res
