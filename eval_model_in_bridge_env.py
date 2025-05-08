"""
eval_model_in_bridge_env.py

Runs a model checkpoint in a real-world Bridge V2 environment.

Usage:
    # VLA:
    python experiments/robot/bridge/eval_model_in_bridge_env.py \
        --model.type <VLM_TYPE> \
        --pretrained_checkpoint <CHECKPOINT_PATH>

    # Octo:
    python experiments/robot/bridge/eval_model_in_bridge_env.py --model_family octo \
         --blocking True --control_frequency 2.5

    # RT-1-X:
    python experiments/robot/bridge/eval_model_in_bridge_env.py --model_family rt_1_x \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import cv2
import draccus
import numpy as np

from prismatic.conf import ModelConfig, ModelRegistry

# TODO (@moojink) Hack so that the interpreter can find experiments.robot
sys.path.append("../..")
from experiments.bridge.utils import (
    draw_bboxes,
    draw_gripper,
    draw_interactive,
    get_action,
    get_image_resize_size,
    get_model,
    get_next_task_label,
    get_octo_policy_function,
    get_preprocessed_image,
    # get_widowx_env,
    make_reasoning_image,
    refresh_obs,
    save_rollout_gif,
)

import interbotix_common_modules.angle_manipulation as ang
import pyrealsense2 as rs  # camera
import json
from scipy.interpolate import UnivariateSpline
import math

from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image

def get_ee_pose(bot):
    # Get current end-effector pose
    T_sb = bot.arm.get_ee_pose()
    # print(T_sb)
    R, p = T_sb[:3, :3], T_sb[:3, 3]
    x, y, z = p[0], p[1], p[2]
    roll, pitch, yaw = ang.rotationMatrixToEulerAngles(R)

    # Get gripper state
    gripper_value = bot.gripper.gripper_command.cmd
    if gripper_value > 0:
        gripper_state = 1  # open
    else:
        gripper_state = 0  # closed
    state = [x, y, z, roll, pitch, yaw, gripper_state]
    return state

def get_qpos(bot):
    # Get current end-effector pose
    joint_names = bot.arm.core.js_index_map
    return bot.arm.core.joint_states.position 

def new_ee_pose(bot, curr, delta, verbose=False):
    """
    Takes in octo delta and executes command
    Note that grasp is not a delta, but absolute
    curr: list of [x,y,z,roll,pitch,yaw,gripper_state]
    delta: list of [dx,dy,dz,droll,dpitch,dyaw,grasp]
    """
    x, y, z, roll, pitch, yaw, grasp = (
        curr[0],
        curr[1],
        curr[2],
        curr[3],
        curr[4],
        curr[5],
        curr[6],
    )

    dx, dy, dz, droll, dpitch, dyaw, dgrasp = (
        delta[0],
        delta[1],
        delta[2],
        delta[3],
        delta[4],
        delta[5],
        delta[6],
    )

    newx, newy, newz, newroll, newpitch, newyaw = (
        x + dx,
        y + dy,
        z + dz,
        roll + droll,
        pitch + dpitch,
        yaw + dyaw,
    )
    if verbose:
        print("Curr: ", end=" ")
        print(["{:.5f}".format(item) for item in curr])
        print("Delt: ", end=" ")
        print(["{:.5f}".format(item) for item in delta])
        # print(f"Curr: {np.around(curr,5):.5f}")
        # print(f"Delt: {np.around(delta,5):.5f}")

    # Set desired transformation
    T_sd = np.identity(4)
    T_sd[:3, :3] = ang.eulerAnglesToRotationMatrix([newroll, newpitch, newyaw])
    T_sd[:3, 3] = [newx, newy, newz]

    # Go to updated pose
    bot.arm.set_ee_pose_matrix(T_sd)

    # Update gripper?
    epsilon = 0.7
    if dgrasp >= epsilon:
        bot.gripper.open()
        newgrasp = 1
    else:
        bot.gripper.close()
        newgrasp = 0
    
    new = [newx, newy, newz, newroll, newpitch, newyaw, newgrasp]
    if verbose:
        # print(f"Pred: {np.round(new,5):.5f}")
        print("Pred: ", end=" ")
        print(["{:.5f}".format(item) for item in new])

    # Get Actual
    state = get_ee_pose(bot)
    if verbose:
        # print(f"Actu: {np.round(state,5):.5f}")
        print("Actu: ", end=" ")
        print(["{:.5f}".format(item) for item in state])
        error = [x - y for x, y in zip(state, new)]
        # print(f"Error: {np.round(error,5):.5f}")
        print("Errr: ", end=" ")
        print(["{:.5f}".format(item) for item in error])
        if dz > 0:
            print("DZ > 0")
        print("")


def take_picture(pipeline):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()

    color_data = color.as_frame().get_data()
    np_image = np.asanyarray(color_data)

    # Rotate
    np_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)
    np_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)
    return np_image


def init_camera():
    """
    Take a few pictures, as the first few have weird lighting
    """
    pipeline = rs.pipeline()
    pipeline.start()

    for i in range(5):
        frames = pipeline.wait_for_frames()
        color = frames.get_color_frame()
        # depth = frames.get_depth_frame()

        color_data = color.as_frame().get_data()
        np_image = np.asanyarray(color_data)

        # Rotate

        np_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)
        np_image = cv2.rotate(np_image, cv2.ROTATE_90_CLOCKWISE)

    return pipeline

def record_state(bot, curr, actions, step, logger,logfile):
    state = get_ee_pose(bot)

    x, y, z, roll, pitch, yaw, grasp = (curr[0], curr[1], curr[2], curr[3], curr[4], curr[5], curr[6],)
    dx, dy, dz, droll, dpitch, dyaw, dgrasp = (actions[0], actions[1], actions[2], actions[3], actions[4], actions[5], actions[6],)
    new = [x + dx,y + dy,z + dz,roll + droll,pitch + dpitch,yaw + dyaw,grasp]

    logger["ee_state_cmd"][step] = new
    logger["traj_act"][step] = actions # End effector actions
    logger["traj_ee_state"][step] = state
    error = [x - y for x, y in zip(state, new)]
    logger["traj_err"][step] = error # error in end effector state
    qpos = get_qpos(bot)
    logger["qpos_traj"][step] = qpos
    with open(logfile, 'w') as f:
        json.dump(logger, f)
    return new

def sleep(bot):
    bot.arm.go_to_sleep_pose()


def warm_filter(img):
    # We are giving y values for a set of x values.
    # And calculating y for [0-255] x values accordingly to the given range.
    increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 95, 175, 255])(
        range(256)
    )

    middle_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 145, 255])(
        range(256)
    )

    # Similarly construct a lookuptable for decreasing pixel values.
    decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 55, 105, 255])(
        range(256)
    )
    # Split the blue, green, and red channel of the image.
    red_channel, green_channel, blue_channel = cv2.split(img)

    # Increase red channel intensity using the constructed lookuptable.
    red_channel = cv2.LUT(red_channel, increase_table).astype(np.uint8)

    # green_channel = cv2.LUT(green_channel, decrease_table).astype(np.uint8)

    # Decrease blue channel intensity using the constructed lookuptable.
    blue_channel = cv2.LUT(blue_channel, decrease_table).astype(np.uint8)

    # Merge the blue, green, and red channel.
    filterd_img = cv2.merge((red_channel, green_channel, blue_channel))
    return filterd_img

def rotationMatrixToEulerAngles(R):
    """
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(roll, pitch, yaw):
    """
    Convert euler angles to rotation matrix
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)],
        ]
    )
    R_y = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )
    R_z = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(R_z, np.dot(R_y, R_x))



@dataclass
class GenerateConfig:
    # fmt: off

    # ModelConfig from `prisma/conf/models.py`; override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.REPRODUCTION_7B.model_id)
    )
    model_family: str = "llava"                                 # Base VLM model family (for prompt builder)

    # Model Parameters
    pretrained_checkpoint: str ="Embodied-CoT/ecot-openvla-7b-bridge"

    # # Environment-Specific Parameters
    # host_ip: str = "localhost"
    # port: int = 5556

    # # Note (@moojink) =>> Setting initial orientation with a 30 degree offset -- more natural!
    # init_ee_pos: List[float] = field(default_factory=lambda: [0.3, 0., 0.16])
    # init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    # bounds: List[List[float]] = field(default_factory=lambda: [
    #         [0.1, -0.20, -0.01, -1.57, 0],
    #         [0.45, 0.25, 0.30, 1.57, 0],
    #     ]
    # )

    # camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = True
    max_episodes: int = 500
    max_steps: int = 600
    control_frequency: float = 2

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path("/home/iromlab/.cache/huggingface/stored_tokens")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    bot = InterbotixManipulatorXS("wx250s", "arm", "gripper")
    bot.gripper.open()
    sleep(bot)
    time.sleep(1)
    bot.gripper.gripper_value = 300.0
    pipeline = init_camera()
    curr = get_ee_pose(bot)

    # if 'left' in configuration_data["carrot_position"]:
    #     delta = [0.18, 0.06, 0.03, 0, 0.875, 0, 1]
    # elif 'right' in configuration_data["carrot_position"]:
    #     delta = [0.18, -0.03, 0.03, 0, 0.875, 0, 1]
    # elif 'middle' in configuration_data["carrot_position"]:
    #     delta = [0.18, 0.0, 0.03, 0, 0.875, 0, 1]
    # else:
    #     raise ValueError('Invalid carrot position')
    # 
    delta = [0.18, 0.0, 0.03, 0, 0.875, 0, 1]
    # delta = [0.0, 0.0, 0.0, 0, 0.0, 0, 0]
    new_ee_pose(bot, curr, delta)
    set_initial_position = get_ee_pose(bot)

    # Load Model --> Get Expected Image Dimensions
    # model = get_model(cfg)
    device = "cuda"
    # path_to_hf = "Embodied-CoT/ecot-openvla-7b-bridge"
    path_to_hf = "/home/iromlab/.cache/huggingface/hub/models--Embodied-CoT--ecot-openvla-7b-bridge/snapshots/492b3dbf3df380f6da333f86ce06dab028176166"
    processor = AutoProcessor.from_pretrained(path_to_hf, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(path_to_hf, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    resize_size = get_image_resize_size(cfg)

    print(type(vla))
    

    # [Octo] Create JAX JIT-compiled policy function.
    # policy_fn = None
    # if cfg.model_family == "octo":
    #     policy_fn = get_octo_policy_function(model)

   
    trial_directory = "./trial_1/"
    os.makedirs(trial_directory, exist_ok=True)
    
    init_img = take_picture(pipeline)
    init_img_fname = trial_directory + "init_img.jpg"
    plt.imsave(init_img_fname, init_img)


    # Initialize the Widow-X Environment
    # env = get_widowx_env(cfg, model)

    # === Start Evaluation ===
    task_label = ""
    episode_idx = 0
    # prev_action = np.array([0, 0, 0, 0, 0, 0, 1.0])

    while episode_idx < cfg.max_episodes:
        # Get Task Description from User
        task_label = get_next_task_label(task_label)
        prompt = "A chat between a curious user and an artificial intelligence assistant. " + \
            "The assistant gives helpful, detailed, and polite answers to the user's questions. " + \
            f"USER: What action should the robot take to {task_label.lower()}? ASSISTANT: TASK:"
        print(task_label)
        rollout_images = []

        # vla.reset_async()

        # model.reset_async()

        # Reset Environment
        # obs, _ = env.reset()

        # Setup
        t = 0
        zero_action_count = 0
        step_duration = 1.0 / cfg.control_frequency

        # Start Episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the Camera Image and Proprioceptive State
                    print("Taking image...", end=" ")

                    img = take_picture(pipeline)
                    img_fname = trial_directory + "Original" + str(t) + ".jpg"
                    plt.imsave(img_fname, img)
                    img = warm_filter(img)
                    img_fname = trial_directory + "Step" + str(t) + ".jpg"
                    plt.imsave(img_fname, img)
                    # Reload Image as Image File for OpenVLA
                    img = Image.open(img_fname)
                    img_array = np.array(img.resize((640, 480)))
                    print(img_array.shape)
                    if len(img_array.shape) == 4:
                        video_image = img_array[-1]
                    else:
                        video_image = img_array
                    
                    img = Image.open(img_fname).resize((256, 256))


                    # obs = refresh_obs(obs, env)
                    # time.sleep(0.1)
                    # obs = refresh_obs(obs, env)
                    print("done.")

                    # Save Image for Rollout GIF =>> Switch on History / No History
                    

                    # Get Preprocessed Image
                    # obs["full_image"] = get_preprocessed_image(obs, resize_size)

                    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)

                    # Query Model --> Get Action
                    # info_dict = dict()
                    # action = get_action(cfg, model, obs, task_label, policy_fn, info_dict=info_dict)
                    action, generated_ids = vla.predict_action(
                    **inputs, unnorm_key="bridge_orig", max_new_tokens=1024
                    )
                    actions_openVLA = np.array(action)
                    actions_nstep = actions_openVLA
                    decoded_tokens = processor.batch_decode(generated_ids)[0]

                    # Add the reasoning to the image
                    # try:
                    reasoning_img, metadata = make_reasoning_image(decoded_tokens)
                    print(metadata)
                    draw_gripper(video_image, metadata["gripper"])
                    draw_bboxes(video_image, metadata["bboxes"])
                    # draw_interactive(video_image, model.use_interactive)
                    draw_interactive(video_image, True)
                    video_image = np.concatenate([video_image, reasoning_img], axis=1)
                    # except ValueError as e:
                    #     print(e)
                    #     print("\033[93m\033[1mWARNING:\033[0m Can't draw reasoning image.")
                    #     video_image = np.concatenate([video_image, np.zeros_like(video_image)], axis=1)
                    
                    rollout_images.append(video_image)
                    save_rollout_gif(rollout_images, f"{episode_idx}_{task_label.replace(' ', '-')}")

                    if metadata["bboxes"] == {}:
                        continue

                    print()

                    if True:  # has issues with X11 display on dgx
                        bgr_img = cv2.cvtColor(video_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("image.png", bgr_img)

                    # [OpenVLA] End episode early if the robot doesn't move at all for a few consecutive steps!
                    #   - Reason: Inference is pretty slow with a single local GPU...
                    if (
                        cfg.model_family == "llava"
                        and np.isclose(np.linalg.norm(action), 1, atol=0.01)
                        and np.linalg.norm(action[:6]) < 0.01
                    ):
                        zero_action_count += 1
                        if zero_action_count == 11:
                            print("Ending episode early due to robot inaction.")
                            break
                    else:
                        zero_action_count = 0

                    # Execute Action
                    print("action:", action)
                    t += 1
                    # TODO: If action is malformed, i.e. not all 7 elements are action tokens, repeat the step
                    # obs, _, _, _, _ = env.step(action)
                    for i in range(1):
                        action_curr = actions_nstep
                        # action_curr[0] = np.clip(action_curr[0], -0.03, 0.01)
                        # action_curr[0] = np.clip(action_curr[0], -0.01, 0.01)
                        
                        actions = action_curr.tolist()
                        # action_history.append(actions)
                        # with open(action_history_fname, "wb") as f:
                        #     pickle.dump(action_history, f)

                        curr_rpy = rotationMatrixToEulerAngles(bot.arm.T_sb[:3, :3])
                        currx = bot.arm.T_sb[0,3]
                        curry = bot.arm.T_sb[1,3]
                        currz = bot.arm.T_sb[2,3]
                        currroll = curr_rpy[0]
                        currpitch = curr_rpy[1]
                        curryaw = curr_rpy[2]

                        dx = actions[0]
                        dy = actions[1]
                        dz = actions[2]
                        droll = actions[3]
                        dpitch = actions[4]
                        dyaw = actions[5]
                        dgrasp = actions[6]

                        print()
                        print(f"dx: {dx}")
                        print(f"dy: {dy}")
                        print(f"dz: {dz}")
                        print(f"droll: {droll}")
                        print(f"dpitch: {dpitch}")
                        print(f"dyaw: {dyaw}")
                        print(f"Gripper: {dgrasp}")
                        print("")

                        moving_time=1.0
                        
                        bot.arm.set_ee_pose_components(
                            x = currx+dx,
                            y = curry+dy,
                            z = currz+dz,
                            roll=currroll+droll,
                            pitch = currpitch + dpitch,
                            yaw = curryaw + dyaw,
                            moving_time = moving_time,
                            blocking=True,
                            custom_guess = bot.arm.get_joint_commands(),
                        )

                        epsilon = 0.7
                        if dgrasp >= epsilon:
                            bot.gripper.open()
                        else:
                            bot.gripper.close()
                            
                        # step += 1

                        # if not useOcto:
                        #     time.sleep(1.0)

                        # if use_logger:
                        #     curr = record_state(bot, curr,actions,step,logger,loggerfile)

                        # Take Picture for storage frame by frame; the actual image for eval is taken at the start of the while loop
                        # if not (i == n_steps - 1):
                        #     img = take_picture(pipeline)
                        #     img_fname = trial_directory + "Original" + str(step) + ".jpg"
                        #     plt.imsave(img_fname, img)

                        #     with open(img_history_fname, "wb") as f:
                        #         pickle.dump(img_history, f)
                    # prev_action = action
            
            except (KeyboardInterrupt, Exception) as e:
                save_rollout_gif(rollout_images, f"{episode_idx}_{task_label.replace(' ', '-')}")
                bot.gripper.open()
                sleep(bot)
                time.sleep(1)
                bot.gripper.gripper_value = 300.0
                curr = get_ee_pose(bot)

                # if 'left' in configuration_data["carrot_position"]:
                #     delta = [0.18, 0.06, 0.03, 0, 0.875, 0, 1]
                # elif 'right' in configuration_data["carrot_position"]:
                #     delta = [0.18, -0.03, 0.03, 0, 0.875, 0, 1]
                # elif 'middle' in configuration_data["carrot_position"]:
                #     delta = [0.18, 0.0, 0.03, 0, 0.875, 0, 1]
                # else:
                #     raise ValueError('Invalid carrot position')
                # 
                delta = [0.18, 0.0, 0.03, 0, 0.875, 0, 1]
                # delta = [0.0, 0.0, 0.0, 0, 0.0, 0, 0]
                new_ee_pose(bot, curr, delta)
                set_initial_position = get_ee_pose(bot)
                continue_flag = input("continue? (y/n)")
                if "y" in continue_flag:
                    break                    
                else:
                    exit()

            # except (KeyboardInterrupt, Exception) as e:
            #     if isinstance(e, KeyboardInterrupt):
            #         print("\nCaught KeyboardInterrupt")
            #         while True:
            #             print("Press Enter to use interactive mode, or type 'exit' to terminate.")
            #             request = input()
            #             if request in ["exit", "continue", ""]:
            #                 break
            #             else:
            #                 continue

            #         if request == "":
            #             # model.use_interactive = True
            #             continue
            #         elif request == "continue":
            #             continue
            #     else:
            #         print(f"\nCaught exception: {e}")
            #         traceback.print_exception(type(e), e, e.__traceback__)
            #         print("")

            #     break

        # Save a Replay GIF of the Episode
        save_rollout_gif(rollout_images, f"{episode_idx}_{task_label.replace(' ', '-')}")

        # Redo Episode or Continue...
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
