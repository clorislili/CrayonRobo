import os
import sys
import shutil
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import numpy as np
import torch
# import open3d as o3d
import torch.nn.functional as F
# import utils
# from utils import get_global_position_from_camera
from sapien.core import Pose
from env_ori import Env,ContactError
from camera import Camera
from robots.panda_robot import Robot
import imageio
import cv2
import json
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import llama
import math
def plot_heatmap(data,savepath,mask):
    """
    Helper function to plot data with associated colormap.
    """
    n = 1
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    ax=axs.flat[0]
    ax.xaxis.set_ticks_position('top')  
    ax.invert_yaxis()
    psm = ax.pcolormesh(data, cmap='RdBu_r', rasterized=True, vmin=-1, vmax=1)#'hsv''gist_ncar''rainbow''jet'
    fig.colorbar(psm, ax=ax)
    plt.savefig(os.path.join(out_dir, savepath))
def distance_between_points(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

parser = ArgumentParser()
parser.add_argument('--llama_dir', type=str, help='llama directory')
parser.add_argument('--adapter_dir', type=str,default='./', help='adapter directory')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if out_dir exists [default: False]')

parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--record_name', type=str)
parser.add_argument('--out_dir', type=str)
eval_conf = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')
print('Loading ckpt....')


# setup env: camera and object and state and blah are strictly follow the status of collecting (refer checkcollect_data.py to reproduce)
#previous info are saved in result.json
shape_id, category, cnt_id, primact_type, trial_id = eval_conf.record_name.split('_')

out_dir = os.path.join(eval_conf.out_dir, '%s_%s_%s_%s_%d' % (shape_id, category, cnt_id, primact_type, int(trial_id)))


flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict()
try:
    print(os.path.join(eval_conf.data_dir, eval_conf.record_name))
    with open(os.path.join(eval_conf.data_dir, eval_conf.record_name, 'result.json'), 'r') as fin:
        replay_data = json.load(fin)
except:
    exit(1)


env = Env()

# setup camera
cam_theta = replay_data['camera_metadata']['theta']
cam_phi = replay_data['camera_metadata']['phi']
cam_dist = replay_data['camera_metadata']['dist']
cam = Camera(env, image_size=336,dist=cam_dist,theta=cam_theta, phi=cam_phi)


# load shape
object_urdf_fn = '../data_collection/assets/%s/mobility.urdf' % shape_id
object_material = env.get_material(4, 4, 0.01)
scale = replay_data['scale']
state = replay_data['object_state']
print( 'Object State: %s' % state)
env.load_object(object_urdf_fn, object_material, scale,state=state)
env.set_object_joint_angles(replay_data['joint_angles'])
cur_qpos = env.get_object_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 5000 and wait_timesteps < 20000:
    env.step()
    env.render()
    cur_new_qpos = env.get_object_qpos()
    invalid_contact = False
    for c in env.scene.get_contacts():
        for p in c.points:
            if abs(p.impulse @ p.impulse) > 1e-4:
                invalid_contact = True
                break
        if invalid_contact:
            break
    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1

if still_timesteps < 5000:
    print('Object Not Still!')
    env.close()
    exit(1)

#use camera to capture
rgb, depth = cam.get_observation()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb_img.png'))
img = Image.fromarray((rgb*255).astype(np.uint8))
gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
mask = (gt_movable_link_mask > 0)

#obtain  movable pixels
xs, ys = np.where(gt_movable_link_mask > 0)
if len(xs) == 0:
    env.scene.remove_articulation(env.object)
    print("cant find ctpt")
    flog.write('cant find ctpt')
    flog.close()
    env.close()
    with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
        json.dump(out_info, fout)
    exit(2)

if eval_conf.adapter_dir == './':
    if os.path.exists(os.path.join(out_dir, 'prediction.json')):
        with open(os.path.join(out_dir, 'prediction.json'), 'r') as fin:
            result = json.load(fin)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!no prediction !!!!!!!!!!!!!!!!!!!!!!!!')
        flog.close()
        env.close()
        exit(2)


print('Answer from model: ', result)
object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
x, y = result.split('(')[1].split(')')[0].split(', ')
x = int(x)
y = int(y)


norm_dir = gt_nor[x,y]

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))


d_x, d_y, d_z = result.split('[')[1].split(']')[0].split(', ')
gripper_direction_camera = np.array([int(d_x)*0.02, int(d_y)*0.02, int(d_z)*0.02])

fd_x, fd_y, fd_z = result.split('[')[2].split(']')[0].split(', ')
gripper_forward_direction_camera = np.array([int(fd_x)*0.02, int(fd_y)*0.02, int(fd_z)*0.02])
try:
    md_x, md_y, md_z = result.split('[')[3].split(']')[0].split(', ')
    moving_direction_cam = np.array([int(md_x)*0.02, int(md_x)*0.02, int(md_x)*0.02])
    moving_direction_cam /= np.linalg.norm(moving_direction_cam)
except:
    flog.close()
    env.close()
    exit()

draw = ImageDraw.Draw(img)
draw.point((y,x),'red')
img.save(os.path.join(out_dir, 'contact_point.png'))

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
position_cam = cam_XYZA[x, y, :3]

position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
target_part_id = object_link_ids[gt_movable_link_mask[x, y] - 1]
env.set_target_object_part_actor_id(target_part_id)
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id


def plot_mani(cam,up, forward,moving_direction_cam):
    if primact_type == 'pushing':
        if (up @ norm_dir[:3] ) > 0:
            up = -up
    elif primact_type == 'pulling':
        if (up @ norm_dir[:3] ) > 0:
            up = -up
    
    up /= np.linalg.norm(up)
    up_cam = up
    left_cam = forward
    left_cam /= np.linalg.norm(left_cam)
    forward_cam = np.cross(left_cam, up_cam)
    forward_cam /= np.linalg.norm(forward_cam)

    up = cam.get_metadata()['mat44'][:3,:3] @ up
    forward = cam.get_metadata()['mat44'][:3,:3] @ forward
    out_info['gripper_direction_world'] = up.tolist()
    
    up = np.array(up, dtype=np.float32)
    up /= np.linalg.norm(up)
    
    left = np.array(forward, dtype=np.float32)
    left /= np.linalg.norm(left)

    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)

    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    final_dist = 0.11
    mid_rotmat = np.array(rotmat, dtype=np.float32)
    mid_rotmat[:3, 3] = position_world - up * final_dist
    mid_pose = Pose().from_transformation_matrix(mid_rotmat)

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - up * 0.15
    start_pose = Pose().from_transformation_matrix(start_rotmat)
    


    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)
    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))
    
    # move back
    robot.open_gripper()
    robot.robot.set_root_pose(start_pose)
    env.render()
    rgb_final_pose, _ = cam.get_observation()
    Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_start_pose.png'))


    out_info['start_target_part_qpos'] = env.get_target_part_qpos()

    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
    touch_position_world_xyz_start = position_world_xyz1[:3]
    success = True

    

        # approach
    robot.move_to_target_pose(mid_rotmat, 2000)
    robot.close_gripper()
    robot.wait_n_steps(2000)
    rgb_final_pose, _ = cam.get_observation()
    Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_mid_pose.png'))
    move_dir = None
    move_left = np.dot(abs(left_cam),abs(moving_direction_cam))
    norm_a = np.linalg.norm(left_cam)
    norm_b = np.linalg.norm(moving_direction_cam)
    cosine_similarity_left = move_left / (norm_a * norm_b)

    move_up = np.dot(abs(up_cam),abs(moving_direction_cam))
    norm_a = np.linalg.norm(up_cam)
    norm_b = np.linalg.norm(moving_direction_cam)
    cosine_similarity_up = move_up / (norm_a * norm_b)

    move_for = np.dot(abs(forward_cam),abs(moving_direction_cam))
    norm_a = np.linalg.norm(forward_cam)
    norm_b = np.linalg.norm(moving_direction_cam)
    cosine_similarity_for = move_for / (norm_a * norm_b)
    


    move_dir = None
    
    if primact_type == 'pulling':
        if abs(cosine_similarity_left) > cosine_similarity_up or cosine_similarity_left>0.8:
            move_dir = 'left'
            final_rotmat = np.array(rotmat, dtype=np.float32)
            final_rotmat[:3, 3] = position_world - left * 0.15
            final_pose = Pose().from_transformation_matrix(final_rotmat)
            
            robot.close_gripper()
            robot.move_to_target_pose(final_rotmat, 2000)
            robot.wait_n_steps(2000)
        elif abs(cosine_similarity_for) > cosine_similarity_up or cosine_similarity_for>0.8:
            move_dir = 'for'
            final_rotmat = np.array(rotmat, dtype=np.float32)
            final_rotmat[:3, 3] = position_world - forward * 0.15
            final_pose = Pose().from_transformation_matrix(final_rotmat)
            
            robot.close_gripper()
            robot.move_to_target_pose(final_rotmat, 2000)
            robot.wait_n_steps(2000)
        else:        
            
            move_dir = 'up'
            final_rotmat = np.array(rotmat, dtype=np.float32)
            final_rotmat[:3, 3] = position_world - up * 0.15
            final_pose = Pose().from_transformation_matrix(final_rotmat)
            
            robot.close_gripper()
            robot.move_to_target_pose(final_rotmat, 2000)
            robot.wait_n_steps(2000)
        out_info['target_rotmat_world'] = final_rotmat.tolist()


    
    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
    out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
    out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()
    if success:
        out_info['result'] = 'VALID'
        out_info['final_target_part_qpos'] = env.get_target_part_qpos()
        if primact_type == 'pushing':
            abs_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])
            j = out_info['target_object_part_joint_id']
            tot_motion = replay_data['joint_angles_upper'][j] - replay_data['joint_angles_lower'][j] + 1e-8
            mani_success = (abs_motion > 0.01) or (abs_motion / tot_motion > 0.5)
            out_info['mani_succ'] = str(mani_success)
        elif primact_type == 'pulling':
            abs_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])
            j = out_info['target_object_part_joint_id']
            tot_motion = replay_data['joint_angles_upper'][j] - replay_data['joint_angles_lower'][j] + 1e-8
            mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
                            np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32)
            mov_dir /= np.linalg.norm(mov_dir)
            
            if move_dir =='up':
                intended_dir = -np.array(up, dtype=np.float32)
            elif move_dir =='left':
                intended_dir = -np.array(left, dtype=np.float32)
            else:
                intended_dir = -np.array(forward, dtype=np.float32)
            
            success = ((abs_motion > 0.01) or (abs_motion / tot_motion > 0.5) )
            mani_success = ((abs_motion > 0.01) or (abs_motion / tot_motion > 0.5) ) and (intended_dir @ mov_dir > 0.5)
            out_info['mani_succ'] = str(mani_success)
            
    return move_dir, success, mani_success

move_dir, success, mani_succ = plot_mani(cam,gripper_direction_camera, gripper_forward_direction_camera,moving_direction_cam)
out_info['succ'] = np.array(success, dtype=bool).tolist()
 
out_info['mani_succ'] = np.array(mani_succ, dtype=bool).tolist()
rgb_final_pose, _ = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose.png'))

print('----------------------------Manipulation result for task {} is: {}'.format(out_dir.split('/')[-1],mani_succ))
with open(os.path.join(out_dir, 'result_pred.json'), 'w') as fout:
    json.dump(out_info, fout)

flog.close()
env.close()
