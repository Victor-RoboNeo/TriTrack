"""Shared CSV motion loading for csv_to_npz.py and batch_csv_to_npz.py."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_slerp,
)

RootRotation = Literal["quat_xyzw", "euler_xyz_deg"]


class MotionLoader:
    """Load motion CSV: root position, root orientation, then joint DOFs per row."""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        *,
        csv_skip_header: bool = False,
        csv_drop_frame_column: bool = False,
        root_rotation: RootRotation = "quat_xyzw",
        joint_angles_in_degrees: bool = False,
        position_scale: float = 0.01,  # convert cm to m
        expected_dof_count: int | None = None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.csv_skip_header = csv_skip_header
        self.csv_drop_frame_column = csv_drop_frame_column
        self.root_rotation: RootRotation = root_rotation
        self.joint_angles_in_degrees = joint_angles_in_degrees
        self.position_scale = position_scale
        self.expected_dof_count = expected_dof_count
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self) -> None:
        header = 1 if self.csv_skip_header else 0
        if self.frame_range is not None:
            skiprows = header + (self.frame_range[0] - 1)
            max_rows = self.frame_range[1] - self.frame_range[0] + 1
            motion = np.loadtxt(self.motion_file, delimiter=",", skiprows=skiprows, max_rows=max_rows)
        elif self.csv_skip_header:
            motion = np.loadtxt(self.motion_file, delimiter=",", skiprows=1)
        else:
            motion = np.loadtxt(self.motion_file, delimiter=",")

        motion = torch.from_numpy(motion).to(torch.float32).to(self.device)
        if self.csv_drop_frame_column:
            motion = motion[:, 1:]

        self.motion_base_poss_input = motion[:, :3] * self.position_scale

        if self.root_rotation == "quat_xyzw":
            self.motion_base_rots_input = motion[:, 3:7]
            self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw -> wxyz
            dof_start = 7
        elif self.root_rotation == "euler_xyz_deg":
            euler = motion[:, 3:6]
            ex = torch.deg2rad(euler[:, 0])
            ey = torch.deg2rad(euler[:, 1])
            ez = torch.deg2rad(euler[:, 2])
            self.motion_base_rots_input = quat_from_euler_xyz(ex, ey, ez)
            dof_start = 6
        else:
            raise ValueError(f"Unknown root_rotation: {self.root_rotation}")

        self.motion_dof_poss_input = motion[:, dof_start:]
        if self.joint_angles_in_degrees:
            self.motion_dof_poss_input = torch.deg2rad(self.motion_dof_poss_input)

        if self.expected_dof_count is not None and self.motion_dof_poss_input.shape[1] != self.expected_dof_count:
            raise ValueError(
                f"DOF column count {self.motion_dof_poss_input.shape[1]} != expected {self.expected_dof_count}"
            )

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self) -> None:
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(
            index_0 + 1,
            torch.tensor(self.input_frames - 1, device=self.device),
        )
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self) -> None:
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        bool,
    ]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag

    def iter_states(self):
        """Yield one timestep per output frame (used by batch_csv_to_npz)."""
        for _ in range(self.output_frames):
            state = (
                self.motion_base_poss[self.current_idx : self.current_idx + 1],
                self.motion_base_rots[self.current_idx : self.current_idx + 1],
                self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
                self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
                self.motion_dof_poss[self.current_idx : self.current_idx + 1],
                self.motion_dof_vels[self.current_idx : self.current_idx + 1],
            )
            self.current_idx += 1
            yield state
