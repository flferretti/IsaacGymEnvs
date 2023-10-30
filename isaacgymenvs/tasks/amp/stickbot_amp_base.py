# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from ..base.vec_task import VecTask


NUM_OBS = 71 + 3 + 1  # + 1  # 1 + 4 + 6 + 23 + 23  # + 12
NUM_ACTIONS = 23

KEY_BODY_NAMES = ["r_forearm", "l_forearm", "r_sole", "l_sole"]


class StickbotAMPBase(VecTask):
    def __init__(
        self,
        config,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        dt = self.cfg["sim"]["dt"]
        # TODO: check if this is correct, in the config file it is was 2!
        self.dt = self.control_freq_inv * dt

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        dof_torques_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        self.dof_torques_tensor = gymtorch.wrap_tensor(dof_torques_tensor).view(
            self.num_envs, self.num_dof
        )

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        self._previous_root_states = self._root_states.clone()

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(
            self._dof_pos, device=self.device, dtype=torch.float
        )

        self._initial_dof_vel = torch.zeros_like(
            self._dof_vel, device=self.device, dtype=torch.float
        )

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(
            self.num_envs, self.num_bodies, 3
        )

        self._terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.output = torch.zeros(
            self.num_envs, self.num_actions, device=self.device, dtype=torch.float
        )

        self.actions = torch.zeros(
            self.num_envs, self.num_actions, device=self.device, dtype=torch.float
        )

        self.task_reset = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # random targets
        x_random = torch_rand_float(3, 4, (self.num_envs, 1), device=self.device)
        y_random = torch_rand_float(-1, 1, (self.num_envs, 1), device=self.device)
        z_random = torch.ones((self.num_envs, 1), device=self.device) * 0.7
        # z_random = torch_rand_float(0.7, 3, (self.num_envs, 1), device=self.device)

        # z_random = (z_random < 2) * 0.7 + (z_random >= 2) * z_random

        self.tar = torch.cat((x_random, y_random, z_random), dim=1)
        x_next = torch_rand_float(3, 4, (self.num_envs, 1), device=self.device)
        y_next = torch_rand_float(-1, 1, (self.num_envs, 1), device=self.device)
        z_next = torch.zeros((self.num_envs, 1), device=self.device)
        self.tar_next = self.tar + torch.cat((x_next, y_next, z_next), dim=1)

        self.des_vel = torch.ones(self.num_envs, device=self.device) * 1.2

        # self.energy_condition = torch.ones(
        #     self.num_envs, dtype=torch.long, device=self.device
        # )

        rigid_body_states_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rigid_body_states_tensor).view(
            self.num_envs, self.num_bodies, -1
        )

        if self.viewer != None:
            self._init_camera()

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._create_ground_plane()
        self._create_envs(self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        # self.energy_condition[env_ids] = ((torch_rand_float(
        #     0, 1, (len(env_ids),1), device=self.device
        # ) > 0.5) * 1).squeeze()
        # self.thrust_all = torch.zeros(
        #     self.num_envs, 4, dtype=torch.float, device=self.device
        # )
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # TODO: removed for now, need to test
        plane_params.static_friction = 3  # self.plane_static_friction
        plane_params.dynamic_friction = 3  # self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../../assets"
        )
        asset_file = "urdf/stickbot.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        dof_props = self.gym.get_asset_dof_properties(humanoid_asset)

        for i in range(self.num_dof):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][i] = 40
            dof_props["damping"][i] = 20

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.80, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 1, 0)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            handle = self.gym.create_actor(
                env_ptr, humanoid_asset, start_pose, "humanoid", i, 1, 0
            )

            self.gym.set_actor_dof_properties(env_ptr, handle, dof_props)
            # self.gym.enable_actor_dof_force_sensors(env_ptr, handle)
            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

        self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

        for j in range(self.num_dof):
            if dof_props["lower"][j] > dof_props["upper"][j]:
                self.dof_limits_lower.append(dof_props["upper"][j])
                self.dof_limits_upper.append(dof_props["lower"][j])
            else:
                self.dof_limits_lower.append(dof_props["lower"][j])
                self.dof_limits_upper.append(dof_props["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)

        # if self._pd_control:
        self._build_pd_action_offset_scale()

        return

    def _build_pd_action_offset_scale(self):
        num_joints = self.num_dof

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            curr_low = lim_low[j]
            curr_high = lim_high[j]
            curr_mid = 0.1 * (curr_high + curr_low)

            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = 0.8 * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[j] = curr_low
            lim_high[j] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, root_state, prev_root_state, target):
        # print("rew", compute_target_reward(root_state, target))
        self.rew_buf[:], self.task_reset[:] = compute_target_reward(
            root_state, prev_root_state, self.des_vel, self.dt, target
        )
        return

    def _compute_reset(self):
        (
            self.reset_buf[:],
            self._terminate_buf[:],
            _,
        ) = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._dof_pos,
            self._dof_vel,
            self._root_states,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_height,
        )
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        task_obs = self._compute_task_obs(env_ids)
        if env_ids is None:
            # energy_condition = self.energy_condition.reshape(-1, 1)
            self.obs_buf[:] = torch.cat([obs, task_obs], dim=-1)
        else:
            # energy_condition = self.energy_condition[env_ids].reshape(-1, 1)
            self.obs_buf[env_ids] = torch.cat([obs, task_obs], dim=-1)
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._root_states
            tar_pos = self.tar
            tar_pos_next = self.tar_next
            des_vel = self.des_vel
        else:
            root_states = self._root_states[env_ids]
            tar_pos = self.tar[env_ids]
            tar_pos_next = self.tar_next[env_ids]
            des_vel = self.des_vel[env_ids]
        obs = self._compute_location_obs(root_states, tar_pos_next, tar_pos, des_vel)
        return obs

    def _compute_location_obs(self, root_states, tar_pos, tar_pos_next, des_vel):
        root_pos = root_states[:, :3]
        root_rot = root_states[:, 3:7]
        dist_vec = tar_pos - root_pos
        local_tar_in_3d = quat_rotate_inverse(root_rot, dist_vec) / 5
        dist_vec_next = tar_pos_next - root_pos     
        local_tar_in_3d_next = quat_rotate_inverse(root_rot, dist_vec_next)

        obs = torch.cat([local_tar_in_3d, des_vel.reshape(-1, 1)], dim=-1)

        return obs

    def _compute_humanoid_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

        obs = compute_robot_observations(
            root_states, dof_pos, dof_vel, key_body_pos, self._local_root_obs
        )
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        alpha = 0.5
        self.output += alpha * (self.actions - self.output)

        joint_targets = self.output#[:, : self.num_dof]

        # position control
        pd_tar = self._action_to_pd_targets(joint_targets)
        tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, tar_tensor)

        return

    def draw_heading(self, root_states, prev_root_states):
        root_pos = root_states[:, :3]
        root_rot = root_states[:, 3:7]
        root_vel = root_states[:, 7:10]
        heading_rot = calc_heading_quat(root_rot)
        facing_dir = torch.zeros_like(root_pos)
        facing_dir[..., 0] = -1.0
        facing_dir = quat_rotate(heading_rot, facing_dir)
        p1 = gymapi.Vec3(root_pos[0, 0], root_pos[0, 1], root_pos[0, 2])
        p2 = root_pos + facing_dir
        p2 = gymapi.Vec3(p2[0, 0], p2[0, 1], p2[0, 2])
        color = gymapi.Vec3(0.0, 0.0, 0.0)
        gymutil.draw_line(
            p1,
            p2,
            color,
            self.gym,
            self.viewer,
            self.envs[0],
        )

        p3 = root_pos + root_vel
        p3 = gymapi.Vec3(p3[0, 0], p3[0, 1], p3[0, 2])
        color2 = gymapi.Vec3(1.0, 0.0, 0.0)
        gymutil.draw_line(
            p1,
            p3,
            color2,
            self.gym,
            self.viewer,
            self.envs[0],
        )

        diff_vel = root_pos - prev_root_states[:, :3]
        diff_vel = diff_vel / self.dt
        p4 = root_pos + diff_vel
        p4 = gymapi.Vec3(p4[0, 0], p4[0, 1], p4[0, 2])
        color3 = gymapi.Vec3(0.0, 1.0, 0.0)
        gymutil.draw_line(
            p1,
            p4,
            color3,
            self.gym,
            self.viewer,
            self.envs[0],
        )

        self.draw_targets()

    def post_physics_step(self):
        self.progress_buf += 1
        if self.viewer and self.enable_viewer_sync:
             self.gym.clear_lines(self.viewer)
             self.draw_targets()
        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self._root_states, self._previous_root_states, self.tar)
        self._compute_reset()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        task_res_ids = self.task_reset.nonzero(as_tuple=False).squeeze(-1)

        if len(task_res_ids) > 0:
            self.advance_target(task_res_ids)
            if task_res_ids[0] == 0 and self.viewer and self.enable_viewer_sync:
                print("target preso!")
                self.draw_targets()

        if len(env_ids) > 0:
            self.reset_targets(env_ids)
            if env_ids[0] == 0 and self.viewer and self.enable_viewer_sync:
                self.draw_targets()

        self.extras["terminate"] = self._terminate_buf

        self._previous_root_states = self._root_states.clone()

        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def draw_targets(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 1, 0))
        sphere_position = gymapi.Vec3(self.tar[0, 0], self.tar[0, 1], self.tar[0, 2])
        sphere_pose = gymapi.Transform(sphere_position, r=None)
        gymutil.draw_lines(
            sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose
        )
        sphere_position_next = gymapi.Vec3(
            self.tar_next[0, 0], self.tar_next[0, 1], self.tar_next[0, 2]
        )
        sphere_pose_next = gymapi.Transform(sphere_position_next, r=None)
        gymutil.draw_lines(
            sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose_next
        )

    def advance_target(self, env_ids):
        self.tar[env_ids, :] = self.tar_next[env_ids, :].clone()

        x_random = torch_rand_float(2, 2, (len(env_ids), 1), device=self.device)
        y_random = torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device)
        z_random = torch_rand_float(0.7, 2.5, (len(env_ids), 1), device=self.device)
        delta_random = torch.cat((x_random, y_random, z_random), dim=1)

        self.tar_next[env_ids, :] = self.tar_next[env_ids, :] + delta_random
        z_random = (z_random < 1.5) * 0.7 + (z_random >= 1.5) * z_random
        z_random = torch.clip(z_random, 0.7, 0.7)
        self.tar_next[env_ids, 2] = z_random.squeeze()
        self.task_reset[env_ids] = 0

    def reset_targets(self, env_ids):
        # random targets
        x_random = torch_rand_float(3, 3, (len(env_ids), 1), device=self.device)
        y_random = torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device)
        # z_random = torch.ones((len(env_ids), 1), device=self.device) * 0.7
        z_random = torch_rand_float(0.7, 2.5, (len(env_ids), 1), device=self.device)
        z_random = (z_random < 1.5) * 0.7 + (z_random >= 1.5) * z_random
        z_random = torch.clip(z_random, 0.7, 0.7)
        self.tar[env_ids, :] = torch.cat((x_random, y_random, z_random), dim=1)
        # self.des_vel = torch.ones(self.num_envs) * 1
        self.task_reset[env_ids] = 0

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        return to_torch(body_ids, device=self.device, dtype=torch.long)

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        return to_torch(body_ids, device=self.device, dtype=torch.long)

    def _action_to_pd_targets(self, action):
        return self._pd_action_offset + self._pd_action_scale * action

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0
        )
        cam_target = gymapi.Vec3(
            self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0
        )
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(
            char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2]
        )

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_robot_observations(
    root_states, dof_pos, dof_vel, key_body_pos, local_root_obs
):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_pos

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_target_reward(root_state, prev_root_state, des_vel, dt, target):
    # type: (Tensor, Tensor, Tensor, float, Tensor) -> Tuple[Tensor, Tensor]
    root_pos = root_state[:, 0:3]
    root_rot = root_state[:, 3:7]
    root_vel = root_state[:, 7:10]

    pos_err_scale = 0.5
    vel_err_scale = 4.0

    pos_reward_w = 0.5
    vel_reward_w = 0.4
    face_reward_w = 0.1

    # unit vector from root to target
    # target_err = target - root_pos[:, :2]
    # pos_err = torch.sum(torch.square(target_err), dim=-1)
    # pos_err = target_err * target_err
    # target_err = target - root_pos[..., 0:2]
    target_err = target - root_pos
    pos_err = torch.sum(target_err * target_err, dim=-1)
    target_rew = torch.exp(-pos_err_scale * pos_err)

    # # I'm having a reward even if I'm going far from the target
    # # I want to maximize the progress. I'm I'm not going towards the target I should get a negative reward

    dist_pre = torch.linalg.norm(target - prev_root_state[:, 0:3], dim=-1)
    dist_now = torch.linalg.norm(target - root_pos, dim=-1)
    dist_diff = des_vel - (dist_pre - dist_now) / dt
    # dist_diff = torch.clip(dist_diff, 0.0, None)
    dist_diff = torch.square(dist_diff)
    vel_reward = torch.exp(-5 * dist_diff)
    # target_rew = (dist_pre - dist_now) / dt

    # print("target", target[0])
    # print("root_pos", root_pos[0])
    # print("root_pre", prev_root_state[0, :3])
    # print("dist_pre", dist_pre[0])
    # print("dist_now", dist_now[0])

    # print("dist_diff", dist_diff[0])
    # print("target_rew", target_rew[0])

    target_dir = target_err / torch.linalg.norm(target_err)

    vel_root = (root_pos - prev_root_state[:, 0:3]) / dt
    # tar_dir_speed = torch.sum(target_dir * root_vel[..., :2], dim=-1
    tar_dir_speed = torch.sum(target_dir * root_vel, dim=-1)
    # )
    # tar_dir_speed = torch.sum(target_dir * vel_root[..., :2], dim=-1)
    # tar_dir_speed = torch.sum(
    #     vel_root[..., :2] * target_dir, dim=-1
    # )  # the projection of a onto b is a.b/|b|
    tar_vel_err = 1 - tar_dir_speed
    tar_vel_err = torch.clip(tar_vel_err, 0.0, None)
    # vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    heading_rot = calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = -1.0  #! the x axis of the robot is backwards
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(target_dir[..., 0:2] * facing_dir[..., 0:2], dim=-1)
    # facing_err = torch.sum(facing_dir[..., 0:2] * target_dir, dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    # facing_reward = facing_err
    # facing_err = torch.clamp_min(facing_err, 0.0)
    # facing_reward = torch.exp(-facing_err * facing_err)

    dist_mask = pos_err < 0.2  # dist-threshold
    facing_reward[dist_mask] = 2.0
    vel_reward[dist_mask] = 2.0
    target_rew[dist_mask] = 2.0

    target_w = 0.5
    vel_w = 0.4
    facing_w = 0.1

    rew = 0.2 * target_rew + 0.9 * vel_reward + 0.2 * facing_reward

    return rew, dist_mask


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    joint_pos,
    joint_vel,
    root_state,
    max_episode_length,
    enable_early_termination,
    termination_height,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor, Tensor]

    body_height = root_state[:, 2]
    has_fallen = body_height <= termination_height
    too_high = body_height >= 5.0

    ang_vel_penalty = root_state[:, 10:13].pow(2).sum(dim=-1) * 0.001
    # base_lin_vel = quat_rotate_inverse(root_state[:, 3:7], root_state[:, 7:10])
    base_lin_vel = root_state[:, 7:10]
    # print("horiz vel", base_lin_vel[:, 0])
    # velocity tracking reward
    lin_vel_error = torch.square(3 - base_lin_vel[:, 0])
    rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25)

    base_h_err = torch.square(0.63 - body_height)
    rew_base_h = torch.exp(-base_h_err / 0.25)

    # joint vel penalty
    joint_vel_penalty = joint_vel.pow(2).sum(dim=-1) * 1e-3

    reward = (
        -ang_vel_penalty * 0
        + rew_lin_vel_xy * 0.5
        + rew_base_h * 0.1
        - joint_vel_penalty
    )
    reset = has_fallen
    reset = torch.logical_or(reset, too_high)
    reward = -10 * reset
    terminated = progress_buf >= max_episode_length - 1
    reset = reset | terminated
    return reset, terminated, reward
