# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper functions."""

from .utils import (
    resolve_nn_activation,
    split_and_pad_trajectories,
    store_code_state,
    string_to_callable,
    unpad_trajectories,
)
from .normalizer_utils import (
    _build_full_obs_norm_state_from_split,
    freeze_normalizers_on_resume,
    load_and_freeze_full_obs_normalizer,
    load_normalizer_states_on_resume,
    resolve_obs_norm_checkpoint_state,
    save_normalizer_states,
    try_build_rl_split_normalizers_from_checkpoint,
)
