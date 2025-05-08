"""
Main eval agent. Only for Simpler for now.

"""

"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <pretrained_checkpoint> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""
from transformers import AutoModelForVision2Seq, AutoProcessor
import logging
import os
import sys

import cv2
import hydra
import imageio
import numpy as np
import simpler_env
import torch
from prismatic.models import load_vla
from src.model.vla.pizero import PiZeroInference
from src.utils.monitor import log_allocated_gpu_memory, log_execution_time
from accelerate.utils import set_seed
from pathlib import Path
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
sys.path.append("/n/fs/robot-data/openvla")
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_model,
)
from src.agent.bridge_utils import (
    draw_bboxes,
    draw_gripper,
    draw_interactive,
    make_reasoning_image,
    save_rollout_gif,
)
from PIL import Image
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from src.utils.geometry import euler2axangle

log = logging.getLogger(__name__)


def process_action(hw_action):
    action_scale = 1.0
    raw_action = {
        "world_vector": np.array(hw_action[:3]),
        "rotation_delta": np.array(hw_action[3:6]),
        "open_gripper": np.array(hw_action[6:7]),  # range [0, 1]; 1 = open; 0 = close
    }
    # process raw_action to obtain the action to be sent to the maniskill2 environment
    action = {}
    action["world_vector"] = raw_action["world_vector"] * action_scale
    action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

    # Policy outputs roll, pitch, yaw of EE in Space/Base Frame
    roll, pitch, yaw = action_rotation_delta  #
    action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
    action_rotation_axangle = action_rotation_ax * action_rotation_angle
    action["rot_axangle"] = action_rotation_axangle * action_scale

    action["gripper"] = (
        2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
    )  # binarize gripper action to 1 (open) and -1 (close)
    action["terminate_episode"] = np.array([0.0])

    action = np.concatenate(
        [action["world_vector"], action["rot_axangle"], action["gripper"]]
    )

    return action

class EvalAgent:
    def __init__(self, cfg):
        self.n_eval_episode = cfg.n_eval_episode
        self.n_video = cfg.n_video
        self.log_dir = cfg.log_dir
        self.video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        # model
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
        self.load_model(cfg)

        self.cfg = cfg

        if hasattr(cfg, "act_steps"):
            self.act_steps = (
                cfg.act_steps
            )  # e.g., run first two steps of predicted four steps

        # env --- no parallelized
        self.env = simpler_env.make(cfg.env.task)

        # env specifics
        if hasattr(cfg.env, "adapter"):
            self.env_adapter = hydra.utils.instantiate(cfg.env.adapter)

    def load_model(self, cfg):
        self.model = PiZeroInference(cfg, use_ddp=False)
        self.load_checkpoint(cfg.checkpoint_path)
        self.model.freeze_all_weights()
        self.model.to(self.dtype)
        self.model.to(self.device)
        if cfg.get(
            "use_torch_compile", True
        ):  # model being compiled in the first batch which takes some time
            self.model = torch.compile(
                self.model,
                mode="default",  # "reduce-overhead", max-autotune(-no-cudagraphs)
                # backend="inductor", # default: inductor; cudagraphs
            )
        # modes: https://pytorch.org/docs/main/generated/torch.compile.html
        # backends: https://pytorch.org/docs/stable/torch.compiler.html
        self.model.eval()
        log.info(f"Using cuda device: {self.device} dtype: {self.dtype}")
        log_allocated_gpu_memory(log, "loading model")

    def run(self):
        """
        Roughly following simpler_env/simple_inference_visual_matching_prepackaged_envs.py

        Assume no obs history for now
        """
        env = self.env
        env_adapter = self.env_adapter
        cnt_episode = 0
        successes = []

        # run episodes --- not dealing with subtasks
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": cnt_episode,  # this determines the obj inits in bridge
        }
        obs, reset_info = env.reset(options=env_reset_options)
        env_adapter.reset()
        # obs keys: 'agent', 'extra', 'camera_param', 'image'
        # agent: 'qpos', 'qvel', 'eef_pos', 'controller', 'base_pose'
        instruction = env.get_language_instruction()
        recording = self.n_video > 0
        if recording:
            os.environ["TOKENIZERS_PARALLELISM"] = (
                "false"  # avoid tokenizer forking warning about deadlock
            )

            def video_parent_path(x):
                return os.path.join(self.video_dir, f"video_{x}")

            video_writer = imageio.get_writer(video_parent_path(cnt_episode) + ".mp4")
        # is_final_subtask = env.is_final_subtask()
        log.info(
            f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
        )
        # Bridge: {'scene_name': 'bridge_table_1_v1', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': '', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png', 'rgb_overlay_cameras': ['3rd_view_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'episode_model_ids': ['bridge_carrot_generated_modified', 'bridge_plate_objaverse_larger'], 'episode_model_scales': [1.0, 1.0], 'episode_source_obj_name': 'bridge_carrot_generated_modified', 'episode_target_obj_name': 'bridge_plate_objaverse_larger', 'episode_source_obj_init_pose_wrt_robot_base': Pose([0.381995, 0.104536, 0.0175282], [-0.706719, 0.0305475, -0.0305745, -0.706173]), 'episode_target_obj_init_pose_wrt_robot_base': Pose([0.232, -0.047, -0.000468373], [2.00041e-10, -5.10387e-07, -1.6915e-06, -1]), 'episode_id': 5}
        # Fractal: {'scene_name': 'google_pick_coke_can_1_v4', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': 'recolor_tabletop_visual_matching_1', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png', 'rgb_overlay_cameras': ['overhead_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'model_id': 'opened_coke_can', 'model_scale': 1.0, 'distractor_model_ids': None, 'distractor_model_scales': None, 'obj_init_pose_wrt_robot_base': Pose([0.587925, -0.0238302, 0.840576], [0.707052, -0.0081018, -0.01162, -0.70702]), 'orientation': 'laid_vertically'} Instruction: pick coke can Max episode length: 80
        while 1:
            # infer action chunk
            inputs = self.env_adapter.preprocess(env, obs, instruction)
            causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
                self.model.build_causal_mask_and_position_ids(
                    inputs["attention_mask"], dtype=self.dtype
                )
            )
            image_text_proprio_mask, action_mask = (
                self.model.split_full_mask_into_submasks(causal_mask)
            )
            inputs = {
                "input_ids": inputs["input_ids"],
                "pixel_values": inputs["pixel_values"].to(self.dtype),
                "image_text_proprio_mask": image_text_proprio_mask,
                "action_mask": action_mask,
                "vlm_position_ids": vlm_position_ids,
                "proprio_position_ids": proprio_position_ids,
                "action_position_ids": action_position_ids,
                "proprios": inputs["proprios"].to(self.dtype),
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # using bf16 shows ~0.001 difference in action inferred when using vs. not using kv cache (infer_action_naive, needs to pass in full causal_mask instead), if starting from the same initial noise. no difference when using float32
            # https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
            with torch.inference_mode():
                actions = self.model(**inputs)
                # actions_naive = self.model.infer_action_naive(**inputs_naive)
                # print(torch.allclose(actions, actions_naive))
            env_actions = self.env_adapter.postprocess(actions[0].float().cpu().numpy())

            # environment step
            for env_action in env_actions[: self.act_steps]:
                obs, reward, success, truncated, info = env.step(env_action)
                if truncated:
                    break

            # video
            if recording:
                video_writer.append_data(self.env_adapter.get_video_frame(env, obs))

            # update instruction, e.g., pick apple ---> put in top drawer
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction

            # original octo eval only done when timeout, i.e., not upon success
            if truncated:
                successes.append(success)
                if recording:
                    video_writer.close()
                    if success:  # rename video with success
                        os.rename(
                            video_parent_path(cnt_episode) + ".mp4",
                            video_parent_path(cnt_episode) + "_success.mp4",
                        )
                cnt_episode += 1

                # quit
                if cnt_episode >= self.n_eval_episode:
                    break

                # reset
                env_reset_options["obj_init_options"] = {
                    "episode_id": cnt_episode,
                }
                obs, reset_info = env.reset(options=env_reset_options)
                env_adapter.reset()
                instruction = env.get_language_instruction()
                log.info(
                    f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
                )
                recording = self.n_video > cnt_episode
                if recording:
                    video_writer = imageio.get_writer(
                        video_parent_path(cnt_episode) + ".mp4"
                    )

        # summary
        success_rate = np.mean(successes)
        log.info("============ Evaluation Summary ============")
        log.info(f"Number of episodes: {cnt_episode}")
        log.info(f"Success rate: {success_rate}")
        log.info("============================================")

    @log_execution_time(log)
    def load_checkpoint(self, path):
        """load to cpu first, then move to gpu"""
        data = torch.load(path, weights_only=True, map_location="cpu")
        data["model"] = {
            k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
        }  # remove "_orig_mod." prefix if saved model was compiled
        self.model.load_state_dict(data["model"], strict=True)
        log.info(f"Loaded model from {path}")

class EvalAgentOpenVLA(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self, cfg):
        assert cfg.pretrained_checkpoint is not None, (
            "cfg.pretrained_checkpoint must not be None!"
        )
        if "image_aug" in cfg.pretrained_checkpoint:
            assert cfg.center_crop, (
                "Expecting `center_crop==True` because model was trained with image augmentations!"
            )
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), (
            "Cannot use both 8-bit and 4-bit quantization!"
        )

        # Load model
        model = get_model(cfg)

        # [OpenVLA] Check that the model contains the action un-normalization key
        if cfg.model_family == "openvla":
            # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            if (
                cfg.unnorm_key not in model.norm_stats
                and f"{cfg.unnorm_key}_no_noops" in model.norm_stats
            ):
                cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
            assert cfg.unnorm_key in model.norm_stats, (
                f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
            )

        # [OpenVLA] Get Hugging Face processor
        self.processor = None
        if cfg.model_family == "openvla":
            self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = (224, 224)
        self.model = model

    def run(self):
        """
        Roughly following simpler_env/simple_inference_visual_matching_prepackaged_envs.py

        Assume no obs history for now
        """
        env = self.env
        # env_adapter = self.env_adapter
        cnt_episode = 0
        successes = []

        # run episodes --- not dealing with subtasks
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": cnt_episode,  # this determines the obj inits in bridge
        }
        obs, reset_info = env.reset(options=env_reset_options)
        # env_adapter.reset()
        # obs keys: 'agent', 'extra', 'camera_param', 'image'
        # agent: 'qpos', 'qvel', 'eef_pos', 'controller', 'base_pose'
        instruction = env.get_language_instruction()
        recording = self.n_video > 0
        if recording:
            os.environ["TOKENIZERS_PARALLELISM"] = (
                "false"  # avoid tokenizer forking warning about deadlock
            )

            def video_parent_path(x):
                return os.path.join(self.video_dir, f"video_{x}")

            video_writer = imageio.get_writer(video_parent_path(cnt_episode) + ".mp4")
        # is_final_subtask = env.is_final_subtask()
        log.info(
            f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
        )
        # Bridge: {'scene_name': 'bridge_table_1_v1', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': '', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png', 'rgb_overlay_cameras': ['3rd_view_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'episode_model_ids': ['bridge_carrot_generated_modified', 'bridge_plate_objaverse_larger'], 'episode_model_scales': [1.0, 1.0], 'episode_source_obj_name': 'bridge_carrot_generated_modified', 'episode_target_obj_name': 'bridge_plate_objaverse_larger', 'episode_source_obj_init_pose_wrt_robot_base': Pose([0.381995, 0.104536, 0.0175282], [-0.706719, 0.0305475, -0.0305745, -0.706173]), 'episode_target_obj_init_pose_wrt_robot_base': Pose([0.232, -0.047, -0.000468373], [2.00041e-10, -5.10387e-07, -1.6915e-06, -1]), 'episode_id': 5}
        # Fractal: {'scene_name': 'google_pick_coke_can_1_v4', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': 'recolor_tabletop_visual_matching_1', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png', 'rgb_overlay_cameras': ['overhead_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'model_id': 'opened_coke_can', 'model_scale': 1.0, 'distractor_model_ids': None, 'distractor_model_scales': None, 'obj_init_pose_wrt_robot_base': Pose([0.587925, -0.0238302, 0.840576], [0.707052, -0.0081018, -0.01162, -0.70702]), 'orientation': 'laid_vertically'} Instruction: pick coke can Max episode length: 80
        while 1:
            # Get preprocessed image
            img = obs["image"]["3rd_view_camera"]["rgb"]  # (128, 128, 3)
            print(img.shape)
            # reshape to (224, 224, 3)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)

            # Prepare observations dict
            # Note: OpenVLA does not take proprio state as input
            obs = {"full_image": img}

            # using bf16 shows ~0.001 difference in action inferred when using vs. not using kv cache (infer_action_naive, needs to pass in full causal_mask instead), if starting from the same initial noise. no difference when using float32
            # https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
            with torch.inference_mode():
                action = get_action(
                    self.cfg,
                    self.model,
                    obs,
                    instruction,
                    processor=self.processor,
                )
                action = process_action(action)

            env_action = action

            # environment step
            # for env_action in env_actions:
            print(env_action.shape)
            obs, reward, success, truncated, info = env.step(env_action)
            # if truncated:
            #     break

            # video
            if recording:
                video_writer.append_data(get_image_from_maniskill2_obs_dict(env, obs))

            # update instruction, e.g., pick apple ---> put in top drawer
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction

            # original octo eval only done when timeout, i.e., not upon success
            if truncated or success:
                successes.append(success)
                if recording:
                    video_writer.close()
                    if success:  # rename video with success
                        os.rename(
                            video_parent_path(cnt_episode) + ".mp4",
                            video_parent_path(cnt_episode) + "_success.mp4",
                        )
                cnt_episode += 1

                # quit
                if cnt_episode >= self.n_eval_episode:
                    break

                # reset
                env_reset_options["obj_init_options"] = {
                    "episode_id": cnt_episode,
                }
                obs, reset_info = env.reset(options=env_reset_options)
                # env_adapter.reset()
                instruction = env.get_language_instruction()
                log.info(
                    f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
                )
                recording = self.n_video > cnt_episode
                if recording:
                    video_writer = imageio.get_writer(
                        video_parent_path(cnt_episode) + ".mp4"
                    )

        # summary
        success_rate = np.mean(successes)
        log.info("============ Evaluation Summary ============")
        log.info(f"Number of episodes: {cnt_episode}")
        log.info(f"Success rate: {success_rate}")
        log.info("============================================")


class EvalAgentECoT(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_model(self, cfg):
        assert cfg.pretrained_checkpoint is not None, (
            "cfg.pretrained_checkpoint must not be None!"
        )
        if "image_aug" in cfg.pretrained_checkpoint:
            assert cfg.center_crop, (
                "Expecting `center_crop==True` because model was trained with image augmentations!"
            )
        assert not (cfg.load_in_8bit and cfg.load_in_4bit), (
            "Cannot use both 8-bit and 4-bit quantization!"
        )

        # Load model
        # model = get_ecot_model(cfg)
        # model = get_model(cfg)
        self.processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(cfg.pretrained_checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True).to(cfg.device)

        # [OpenVLA] Check that the model contains the action un-normalization key
        # if cfg.model_family == "llava":
            # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
        # if (
        #     cfg.unnorm_key not in model.norm_stats
        #     and f"{cfg.unnorm_key}_no_noops" in model.norm_stats
        # ):
        #     cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, (
            f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
        )

        # [OpenVLA] Get Hugging Face processor
        # self.processor = None
        # if cfg.model_family == "openvla":
        # self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = (224, 224)
        self.model = model


    def run(self):
        """
        Roughly following simpler_env/simple_inference_visual_matching_prepackaged_envs.py

        Assume no obs history for now
        """
        env = self.env
        # env_adapter = self.env_adapter
        cnt_episode = 0
        successes = []

        # run episodes --- not dealing with subtasks
        env_reset_options = {}
        env_reset_options["obj_init_options"] = {
            "episode_id": cnt_episode,  # this determines the obj inits in bridge
        }
        obs, reset_info = env.reset(options=env_reset_options)
        # env_adapter.reset()
        # obs keys: 'agent', 'extra', 'camera_param', 'image'
        # agent: 'qpos', 'qvel', 'eef_pos', 'controller', 'base_pose'
        instruction = env.get_language_instruction()
        instruction = "A chat between a curious user and an artificial intelligence assistant. " + \
            "The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
            f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
        rollout_images = []
        recording = self.n_video > 0
        def video_parent_path(x):
            return os.path.join(self.video_dir, f"video_{x}")
            os.environ["TOKENIZERS_PARALLELISM"] = (
                "false"  # avoid tokenizer forking warning about deadlock
            )

        if recording:
            
            video_writer = imageio.get_writer(video_parent_path(cnt_episode) + ".mp4")
        # is_final_subtask = env.is_final_subtask()
        log.info(
            f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
        )
        # Bridge: {'scene_name': 'bridge_table_1_v1', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': '', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png', 'rgb_overlay_cameras': ['3rd_view_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'episode_model_ids': ['bridge_carrot_generated_modified', 'bridge_plate_objaverse_larger'], 'episode_model_scales': [1.0, 1.0], 'episode_source_obj_name': 'bridge_carrot_generated_modified', 'episode_target_obj_name': 'bridge_plate_objaverse_larger', 'episode_source_obj_init_pose_wrt_robot_base': Pose([0.381995, 0.104536, 0.0175282], [-0.706719, 0.0305475, -0.0305745, -0.706173]), 'episode_target_obj_init_pose_wrt_robot_base': Pose([0.232, -0.047, -0.000468373], [2.00041e-10, -5.10387e-07, -1.6915e-06, -1]), 'episode_id': 5}
        # Fractal: {'scene_name': 'google_pick_coke_can_1_v4', 'scene_offset': None, 'scene_pose': None, 'scene_table_height': 0.87, 'urdf_version': 'recolor_tabletop_visual_matching_1', 'rgb_overlay_path': '.../SimplerEnv/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png', 'rgb_overlay_cameras': ['overhead_camera'], 'rgb_overlay_mode': 'background', 'disable_bad_material': False, 'model_id': 'opened_coke_can', 'model_scale': 1.0, 'distractor_model_ids': None, 'distractor_model_scales': None, 'obj_init_pose_wrt_robot_base': Pose([0.587925, -0.0238302, 0.840576], [0.707052, -0.0081018, -0.01162, -0.70702]), 'orientation': 'laid_vertically'} Instruction: pick coke can Max episode length: 80
        while 1:
            # Get preprocessed image
            raw_img = obs["image"]["3rd_view_camera"]["rgb"]  # (128, 128, 3)
            # reshape to (224, 224, 3)
            img = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            img = Image.fromarray(img.astype(np.uint8))

            # Prepare observations dict
            # Note: OpenVLA does not take proprio state as input
            # obs = {"full_image": img}

            video_image = cv2.resize(raw_img, (640, 480), interpolation=cv2.INTER_LANCZOS4)

            # using bf16 shows ~0.001 difference in action inferred when using vs. not using kv cache (infer_action_naive, needs to pass in full causal_mask instead), if starting from the same initial noise. no difference when using float32
            # https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
            # info_dict = dict()
            with torch.inference_mode():
                # action = get_ecot_action(
                #     self.model,
                #     obs,
                #     instruction,
                #     info_dict=info_dict
                # )
                # action, generated_ids = get_action(
                #     self.cfg,
                #     self.model,
                #     obs,
                #     instruction,
                #     processor=self.processor,
                # )
                inputs = self.processor(instruction, img).to(self.cfg.device, dtype=torch.bfloat16)
                breakpoint()
                action, generated_ids = self.model.predict_action(
                    **inputs, unnorm_key=self.cfg.unnorm_key, max_new_tokens=1024, do_sample=False
                    )
                action = process_action(action)
                breakpoint()
                print(action.shape)
                decoded_tokens = self.processor.batch_decode(generated_ids)[0]
            
            # Add the reasoning to the image
            # try:
            reasoning_img, metadata = make_reasoning_image(decoded_tokens)
            draw_gripper(video_image, metadata["gripper"])
            draw_bboxes(video_image, metadata["bboxes"])
            # draw_interactive(video_image, True)
            video_image = np.concatenate([video_image, reasoning_img], axis=1)
            # except ValueError:
                # print("\033[93m\033[1mWARNING:\033[0m Can't draw reasoning image.")
                # video_image = np.concatenate([video_image, np.zeros_like(video_image)], axis=1)
            rollout_images.append(video_image)

            if True:  # has issues with X11 display on dgx
                bgr_img = cv2.cvtColor(video_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite("image.png", bgr_img)

            env_action = action


            # environment step
            # for env_action in env_actions:
            obs, reward, success, truncated, info = env.step(env_action)
            # if truncated:
            #     break

            # video
            if recording:
                video_writer.append_data(get_image_from_maniskill2_obs_dict(env, obs))

            # update instruction, e.g., pick apple ---> put in top drawer
            new_instruction = env.get_language_instruction()
            new_instruction = "A chat between a curious user and an artificial intelligence assistant. " + \
                "The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
                f"USER: What action should the robot take to {new_instruction.lower()}? ASSISTANT: TASK:"
            if new_instruction != instruction:
                instruction = new_instruction

            # original octo eval only done when timeout, i.e., not upon success
            if truncated or success:
                
                successes.append(success)
                if recording:
                    video_writer.close()
                    if success:  # rename video with success
                        os.rename(
                            video_parent_path(cnt_episode) + ".mp4",
                            video_parent_path(cnt_episode) + "_success.mp4",
                        )
                if success:
                    save_rollout_gif(rollout_images, video_parent_path(cnt_episode) + "_success.mp4")
                else:
                    save_rollout_gif(rollout_images, video_parent_path(cnt_episode) + ".mp4")
                rollout_images = []
                cnt_episode += 1

                # quit
                if cnt_episode >= self.n_eval_episode:
                    break

                # reset
                env_reset_options["obj_init_options"] = {
                    "episode_id": cnt_episode,
                }
                obs, reset_info = env.reset(options=env_reset_options)
                # env_adapter.reset()
                instruction = env.get_language_instruction()
                instruction = "A chat between a curious user and an artificial intelligence assistant. " + \
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
                    f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT: TASK:"
                log.info(
                    f"Reset info: {reset_info} Instruction: {instruction} Max episode length: {env.spec.max_episode_steps}"
                )
                recording = self.n_video > cnt_episode
                if recording:
                    video_writer = imageio.get_writer(
                        video_parent_path(cnt_episode) + ".mp4"
                    )
                

        # summary
        success_rate = np.mean(successes)
        log.info("============ Evaluation Summary ============")
        log.info(f"Number of episodes: {cnt_episode}")
        log.info(f"Success rate: {success_rate}")
        log.info("============================================")
