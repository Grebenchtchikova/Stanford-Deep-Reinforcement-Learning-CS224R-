gsm8k_custom.py:

# Copyright 2025 Individual Contributor: chung
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""
verl-compatible GSM8K reward function used as
    custom_reward_function.path=verl/utils/reward_score/gsm8k_custom.py
verl expects compute_score(data_source, solution_str, ground_truth, extra_info)
but verl.utils.reward_score.gsm8k.compute_score is (solution_str, ground_truth, method, ...).
This module is a thin adapter that hard-codes method='strict' so the reward
signal cannot be satisfied by random numbers leaking in from prepended trivia
facts (Track B). The model MUST emit "#### N" to earn reward.
"""
from verl.utils.reward_score.gsm8k import compute_score as _gsm8k_compute_score
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return _gsm8k_compute_score(
        solution_str=solution_str,
        ground_truth=str(ground_truth),
        method="strict",
        format_score=0.0,
        score=1.0,
    )
