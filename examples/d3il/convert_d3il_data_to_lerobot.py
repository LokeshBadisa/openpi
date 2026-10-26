"""
Minimal example script for converting a dataset to LeRobot format.

We use the D3IL dataset (stored in jpgs & pkls) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/d3il/convert_d3il_data_to_lerobot.py
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
from tqdm import tqdm
import pickle 
from pathlib import Path
from PIL import Image
import numpy as np
import math

BASE_DIR = '/data1/lokesh/d3il/environments/dataset/data'
REPO_NAME = "converted_d3il_1"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "aligning",
    "avoiding",
    "pushing",
    "sorting/2_boxes",
    "sorting/4_boxes",
    "sorting/6_boxes",
    # "stacking",
]  
img_dict = {
    "aligning": 'all_data/high_res',#has both bp & inhand
    "avoiding": 'high_res',
    "pushing": 'high_res',
    "sorting/2_boxes": 'high_res',
    "sorting/4_boxes": 'high_res',
    "sorting/6_boxes": 'high_res',
    "stacking": 'high_res',
}
state_dict = {
    "aligning": 'all_data/state',
    "avoiding": 'data',
    "pushing": 'all_data',
    "sorting/2_boxes": 'state',
    "sorting/4_boxes": 'state',
    "sorting/6_boxes": 'state',
    "stacking": 'all_data',
}
language_instruction_dict = {
    "aligning": "Align the box to the target box position.",
    "avoiding": "Move to go past green line by avoiding colliding with obstacles.",
    "pushing": "Push each box into the target zone. Zone and block need not color-match.",
    "sorting/2_boxes": "Sort the blocks into their color-matching target boxes.",
    "sorting/4_boxes": "Sort the blocks into their color-matching target boxes.",
    "sorting/6_boxes": "Sort the blocks into their color-matching target boxes.",
    "stacking": "Stack the blocks in the target zone."
}

def main():
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME/REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        episode_list = list((Path(f'{BASE_DIR}/{raw_dataset_name}/{state_dict[raw_dataset_name]}').iterdir()))
        for episode in tqdm(episode_list):
            with open(episode, 'rb') as f:
                state = pickle.load(f)
            ep_name = episode.stem
            for step in range(state['robot']['sim_step'].shape[0]-1):
                robot_state = [
                                    state['robot']['c_pos'][step],
                                    _quat2axisangle(state['robot']['c_quat'][step]),                                    
                ]
                if raw_dataset_name == 'stacking':
                    robot_state.append(
                                        np.array([state['robot']['gripper_width'][step],-1.0 * state['robot']['gripper_width'][step]]),
                    )
                
                robot_state = np.concatenate(robot_state, axis=0)

                if raw_dataset_name == 'stacking':
                    if state['robot']['gripper_width'][step+1]>0.075:#0.075 is obtained from L341 of environments/d3il/envs/gym_stacking_env/gym_stacking/envs/stacking.py
                        x = -1.0
                    else:
                        x = 1.0
                

                # We can use delta transform of openpi, therefore not subtracting here with current state                
                robot_action = [
                        state['robot']['des_c_pos'][step+1]-state['robot']['c_pos'][step],
                        _quat2axisangle(state['robot']['des_c_quat'][step+1]) - _quat2axisangle(state['robot']['c_quat'][step])                
                ]
                if raw_dataset_name == 'stacking':
                    robot_action.append(x)

                robot_action = np.concatenate(robot_action, axis=0)
                
                with Image.open(Path(BASE_DIR)/raw_dataset_name/img_dict[raw_dataset_name]/'bp'/ep_name/f"{step}.jpg") as img:
                    image = np.array(img.convert("RGB"))
                with Image.open(Path(BASE_DIR)/raw_dataset_name/img_dict[raw_dataset_name]/'inhand'/ep_name/f"{step}.jpg") as img:
                    wrist_image = np.array(img.convert("RGB"))

                dataset.add_frame(
                    {
                        "image": image,
                        "wrist_image": wrist_image,
                        "state": robot_state.astype(np.float32),
                        "actions": robot_action.astype(np.float32),
                        "task": language_instruction_dict[raw_dataset_name],
                    }
                )
            dataset.save_episode()



def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

if __name__ == "__main__":    
    main()