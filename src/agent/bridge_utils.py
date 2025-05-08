"""Utils for evaluating policies in real-world robot environments."""

import os
import sys
import time
from functools import partial
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from accelerate.utils import set_seed
from PIL import Image

import enum


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_cot_database_keys():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }
sys.path.append("../..")  # hack so that the interpreter can find experiments.robot

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def save_rollout_gif(rollout_images, path):
    """Saves a GIF of an episode."""

    imageio.mimsave(path, rollout_images)
    print(f"Saved rollout GIF at path {path}")
    # Save as mp4
    # mp4_path = f"./rollouts/rollout-{DATE_TIME}-{idx}.mp4"
    # imageio.mimwrite(mp4_path, rollout_images, fps=5)
    # print(f"Saved rollout MP4 at path {mp4_path}")


def resize_image(img, resize_size):
    """Takes numpy array corresponding to a single image and returns resized image as numpy array."""
    assert isinstance(resize_size, tuple)
    img = Image.fromarray(img)
    BRIDGE_ORIG_IMG_SIZE = (256, 256)
    img = img.resize(BRIDGE_ORIG_IMG_SIZE, Image.Resampling.LANCZOS)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # also resize to size seen at train time
    img = img.convert("RGB")
    img = np.array(img)
    return img


def write_text(image, text, size, location, line_max_length):
    next_x, next_y = location

    for line in text:
        x, y = next_x, next_y

        for i in range(0, len(line), line_max_length):
            line_chunk = line[i : i + line_max_length]
            cv2.putText(image, line_chunk, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18

        next_y = max(y, next_y + 50)


def split_reasoning(text, tags):
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                s = v.split(tag)
                new_parts[k] = s[0]
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v

    return new_parts


def get_metadata(reasoning):
    metadata = {"gripper": [[0, 0]], "bboxes": dict()}

    if f" {CotTag.GRIPPER_POSITION.value}" in reasoning:
        gripper_pos = reasoning[f" {CotTag.GRIPPER_POSITION.value}"]
        gripper_pos = gripper_pos.split("[")[-1]
        gripper_pos = gripper_pos.split("]")[0]
        gripper_pos = [int(x) for x in gripper_pos.split(",")]
        gripper_pos = [(gripper_pos[2 * i], gripper_pos[2 * i + 1]) for i in range(len(gripper_pos) // 2)]
        metadata["gripper"] = gripper_pos

    if f" {CotTag.VISIBLE_OBJECTS.value}" in reasoning:
        for sample in reasoning[f" {CotTag.VISIBLE_OBJECTS.value}"].split("]"):
            obj = sample.split("[")[0]
            if obj == "":
                continue
            coords = [int(n) for n in sample.split("[")[-1].split(",")]
            metadata["bboxes"][obj] = coords

    return metadata


def resize_pos(pos, img_size):
    return [(x * size) // 256 for x, size in zip(pos, img_size)]


def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)


def draw_interactive(img, is_interactive):
    if is_interactive:
        cv2.putText(img, "Interactive", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, "Interactive", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]


def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        show_name = name
        # show_name = f'{name}; {str(bbox)}'

        cv2.rectangle(
            img,
            resize_pos((bbox[0], bbox[1]), img_size),
            resize_pos((bbox[2], bbox[3]), img_size),
            name_to_random_color(name),
            2,
        )
        cv2.putText(
            img,
            show_name,
            resize_pos((bbox[0], bbox[1] + 6), img_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def make_reasoning_image(text):
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    tags = [f" {tag}" for tag in get_cot_tags_list()]
    reasoning = split_reasoning(text, tags)

    text = [tag + reasoning[tag] for tag in tags[:-1] if tag in reasoning]
    write_text(image, text, 0.5, (10, 30), 70)

    return image, get_metadata(reasoning)