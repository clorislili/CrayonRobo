"""
    For panda (two-finger) gripper: pushing, pushing-left, pushing-up, pulling, pulling-left, pulling-up
        50% all parts closed, 50% middle (for each part, 50% prob. closed, 50% prob. middle)
        Simulate until static before starting
"""
# import open3d as o3d
import os
import sys
import shutil
import numpy as np
from PIL import Image, ImageDraw
from utils import get_global_position_from_camera, save_h5
# import cv2
import json
from argparse import ArgumentParser
# import sapien.core as sapien
from sapien.core import Pose
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
import math
out_info = dict()
parser = ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--record_name', type=str, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()
def direction_vector(loc1, loc2):
    # loc1 and loc2 are tuples of (x, y) coordinates
    x1, y1 = loc1
    x2, y2 = loc2
    
    # Calculate the vector between loc1 and loc2
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the magnitude of the vector
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    
    # Normalize the vector to get the direction vector
    if magnitude != 0:
        direction_vector = (dx / magnitude, dy / magnitude)
    else:
        direction_vector = (0, 0)  # If the points are the same, return zero vector
        
    return direction_vector
def normalize_vector(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # 计算向量的每个分量
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1

    # 计算向量的长度
    vector_length = math.sqrt(vx**2 + vy**2 + vz**2)

    # 归一化向量
    normalized_vx = vx / vector_length
    normalized_vy = vy / vector_length
    normalized_vz = vz / vector_length

    return (normalized_vx, normalized_vy, normalized_vz)
shape_id, category, cnt_id, primact_type, trial_id = args.record_name.split('_')
out_dir = os.path.join(args.data_dir, args.record_name)
with open(os.path.join(out_dir, 'result.json'), 'r') as fin:
    replay_data = json.load(fin)
def distance_between_points(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance
def direction_vector(loc1, loc2):
    # loc1 and loc2 are tuples of (x, y) coordinates
    x1, y1 = loc1
    x2, y2 = loc2
    
    # Calculate the vector between loc1 and loc2
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate the magnitude of the vector
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    
    # Normalize the vector to get the direction vector
    if magnitude != 0:
        direction_vector = (dx / magnitude, dy / magnitude)
    else:
        direction_vector = (0, 0)  # If the points are the same, return zero vector
        
    return direction_vector


# setup env
env = Env()

# setup camera
cam_theta = replay_data['camera_metadata']['theta']
cam_phi = replay_data['camera_metadata']['phi']
cam_dist = replay_data['camera_metadata']['dist']
cam = Camera(env, theta=cam_theta, phi=cam_phi,image_size=336,dist=cam_dist)

# env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam_theta, -cam_phi)

# load shape
object_urdf_fn = '/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/xcy/where2act/data/(xiaoqi)where2act_original_sapien_dataset/%s/mobility.urdf' % shape_id
object_material = env.get_material(4, 4, 0.01)
scale = replay_data['scale']
state = replay_data['object_state']
print( 'Object State: %s' % state)
env.load_object(object_urdf_fn, object_material, scale,state=state)
# test_dir = f'/vepfs-cnsh4137610c2f4c/algo/user8/lixiaoqi/cloris/where2act-main/result/finger_train_30cats_train_pulling_0416_world_1w_imagelang_movedir_cam_hint014_0421_e9_hint3/{args.record_name}'
# with open(os.path.join(test_dir, 'result_heu_replay_0428.json'), 'r') as fin:
#     test_data = json.load(fin)
env.set_object_joint_angles(replay_data['joint_angles'])
# env.set_object_joint_angles(test_data['joint_qpos'])
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

### use the GT vision
rgb, depth = cam.get_observation()
# with open(os.path.join(out_dir,'depth.json'), "w") as file:
#     json.dump(depth.tolist(), file)
#     exit()
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))
img = Image.fromarray((rgb*255).astype(np.uint8))
draw = ImageDraw.Draw(img)
x,y = replay_data['pixel_locs']
# x_new = x*(960/336)
# y_new = y*(960/336)
# draw.point((y_new,x_new),'red')
# img.save(os.path.join(out_dir, 'contact_point_960.png'))
# exit()
# pull_force =  Image.fromarray(np.array(Image.open(os.path.join(out_dir,'pull_force.png')).convert('RGB')))
# draw = ImageDraw.Draw(pull_force)
position_cam =  np.array(replay_data['position_cam'])
forward_cam = np.array(replay_data['gripper_forward_direction_camera'])
# forward_point2_cam = position_cam + 0.05*forward_cam[:3]
# forward_point2_cam = [-forward_point2_cam[1],-forward_point2_cam[2],forward_point2_cam[0]]
# forward_xy2 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],forward_point2_cam)
# forward_xy2 = (forward_xy2/forward_xy2[2])[:2] 
# forward_vec = direction_vector((forward_xy2[1],forward_xy2[0]),(x,y))
# draw.line([(int(forward_xy2[0]),int(forward_xy2[1])), (y,x)], fill="yellow", width=2)
# pull_force.save(os.path.join(out_dir, 'pull_force_hint4.png'))
out_dict = dict()
# out_dict['forward_vec'] = forward_vec

# rgb_img =  Image.fromarray(np.array(Image.open(os.path.join(out_dir,'rgb.png')).convert('RGB')))
# draw = ImageDraw.Draw(rgb_img)
# pixel_color = "blue"
# radius = 1.5
# # Draw a filled ellipse (pixel)
# draw.ellipse((y - radius, x - radius, 
#               y + radius, x + radius), fill=pixel_color)
# rgb_img.save(os.path.join(out_dir, 'pull_force_hint1.png'))

rgb_img =  Image.fromarray(np.array(Image.open(os.path.join(out_dir,'rgb.png')).convert('RGB')))
draw = ImageDraw.Draw(rgb_img)
action_direction_cam = np.array(replay_data['gripper_direction_camera'])
up_point2 = position_cam - 0.13*action_direction_cam
up_point2 = [-up_point2[1],-up_point2[2],up_point2[0]]
up_xy1 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],up_point2)
up_xy = (up_xy1/up_xy1[2])[:2] # (x,y,r^2,r)
up_2 = (int(up_xy[0]),  int(up_xy[1]))
up_vec = direction_vector((y,x),up_2)

left_cam = np.cross(action_direction_cam, forward_cam)
left_cam /= np.linalg.norm(left_cam)
left_point2_cam = position_cam - 0.13*left_cam[:3]
left_point2_cam = [-left_point2_cam[1],-left_point2_cam[2],left_point2_cam[0]]
left_xy2 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],left_point2_cam)
left_xy2 = (left_xy2/left_xy2[2])[:2] 
left_2 = (int(left_xy2[0]),  int(left_xy2[1]))
# print(direction_vector((y,x),up_2),direction_vector((y,x),left_2))
out_dict['up_dir_new'] = direction_vector((y,x),up_2)
out_dict['left_dir_new'] = direction_vector((y,x),left_2)
pixel_color = "blue"
radius = 3
draw.ellipse((y - radius, x - radius, 
              y + radius, x + radius), fill=pixel_color)
rgb_img.save(os.path.join(out_dir, 'pull_force_hint1.png'))
# exit()
draw.line([(y,x),up_2], fill="red", width=5)
draw.line([(y,x),left_2], fill="green", width=5)


# Draw a filled ellipse (pixel)
draw.ellipse((y - radius, x - radius, 
              y + radius, x + radius), fill=pixel_color)
rgb_img.save(os.path.join(out_dir, 'pull_force_hint3.png'))
# exit()


object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)

# load the pixel to interact
x, y = replay_data['pixel_locs'][0], replay_data['pixel_locs'][1]
env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1])


up = replay_data['gripper_direction_world']
forward = replay_data['gripper_forward_direction_world']
# get pixel 3D position (cam/world)

position_world = replay_data['position_world']
position_world_xyz1 = np.ones(4)
position_world_xyz1[:3] = position_world

# compute final pose
up = np.array(up, dtype=np.float32)
up /= np.linalg.norm(up)
forward = np.array(forward, dtype=np.float32)
left = np.cross(up, forward)
left /= np.linalg.norm(left)
forward = np.cross(left, up)
forward /= np.linalg.norm(forward)
rotmat = np.eye(4).astype(np.float32)
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up

final_dist = 0.11
if primact_type == 'pushing-left' or primact_type == 'pushing-up':
    final_dist = 0.11
# print(position_world , up)
mid_rotmat = np.array(rotmat, dtype=np.float32)
mid_rotmat[:3, 3] = position_world - up * final_dist
mid_pose = Pose().from_transformation_matrix(mid_rotmat)


start_rotmat = np.array(rotmat, dtype=np.float32)
start_rotmat[:3, 3] = position_world - up * 0.15
start_pose = Pose().from_transformation_matrix(start_rotmat)


# _, hinge_point = env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1])
# hinge_quat = hinge_point.q
# hinge_rotmat = env.quaternion_to_rotation_matrix(hinge_quat)
# sim_up = np.dot(hinge_rotmat[:3,2],-up)
# sim_left = np.dot(hinge_rotmat[:3,1],-left)
# print(sim_up,sim_left)
# exit()


pull_rotmat = np.array(rotmat, dtype=np.float32)
pull_rotmat[:3, 3] = position_world - up * (final_dist+0.005)
pull_pose = Pose().from_transformation_matrix(pull_rotmat)



pull_rotmat_left = np.array(rotmat, dtype=np.float32)
pull_rotmat_left[:3, 3] = position_world - left * (final_dist+0.005)
pull_pose_left = Pose().from_transformation_matrix(pull_rotmat_left)



### viz the EE gripper position
# setup robot
# robot = env.load_robot("./robots/xarm6/xarm6_vacuum.urdf")
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))


# move back
robot.open_gripper()
robot.robot.set_root_pose(start_pose)
env.render()



env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in primact_type)

start_target_part_qpos = env.get_target_part_qpos()

target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1
touch_position_world_xyz_start = position_world_xyz1[:3]
success = True

try:

    # approach
    robot.move_to_target_pose(mid_rotmat, 2000)
    robot.close_gripper()
    robot.wait_n_steps(2000)
    rgb_final_pose, _ = cam.get_observation()
    Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_mid_pose_replay.png'))

    move_dir = None
    if primact_type == 'pulling':
        target_link_mat44_before = env.get_target_part_pose().to_transformation_matrix() @ position_local_xyz1
        robot.close_gripper()
        robot.move_to_target_pose(pull_rotmat_left, 2000)
        robot.wait_n_steps(2000)
        target_link_mat44_after = env.get_target_part_pose().to_transformation_matrix() @ position_local_xyz1
        move_left = distance_between_points(target_link_mat44_before[:3],target_link_mat44_after[:3])
        if  move_left > 0.001:
            move_dir = 'left'
            final_rotmat = np.array(rotmat, dtype=np.float32)
            final_rotmat[:3, 3] = position_world - left * 0.15
            final_pose = Pose().from_transformation_matrix(final_rotmat)
            
            robot.close_gripper()
            robot.move_to_target_pose(final_rotmat, 2000)
            robot.wait_n_steps(2000)
        else:
            target_link_mat44_before = env.get_target_part_pose().to_transformation_matrix() @ position_local_xyz1
            robot.close_gripper()
            robot.move_to_target_pose(pull_rotmat, 2000)
            robot.wait_n_steps(2000)
            target_link_mat44_after = env.get_target_part_pose().to_transformation_matrix() @ position_local_xyz1
            
            move_up = distance_between_points(target_link_mat44_before[:3],target_link_mat44_after[:3])
            if move_up > move_left:
                move_dir = 'up'
                final_rotmat = np.array(rotmat, dtype=np.float32)
                final_rotmat[:3, 3] = position_world - up * 0.15
                final_pose = Pose().from_transformation_matrix(final_rotmat)
                robot.close_gripper()
                robot.move_to_target_pose(final_rotmat, 2000)
                robot.wait_n_steps(2000)
            else:
                move_dir = 'left'
                final_rotmat = np.array(rotmat, dtype=np.float32)
                final_rotmat[:3, 3] = position_world - left * 0.15
                final_pose = Pose().from_transformation_matrix(final_rotmat)
                robot.close_gripper()
                robot.move_to_target_pose(final_rotmat, 2000)
                robot.wait_n_steps(2000)
        target_rotmat_world = final_rotmat.tolist()
    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    final_position_world = target_link_mat44 @ position_local_xyz1
    # print(final_position_world,final_rotmat[:3,3])
    
    final_postion_cam =  np.linalg.inv(cam.get_metadata()['mat44']) @ final_position_world
    # print(final_postion_cam)
    final_postion_cam2 = [-final_postion_cam[1],-final_postion_cam[2],final_postion_cam[0]]
    final_pos_xy1 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],final_postion_cam2)
    final_pos_xy = (final_pos_xy1/final_pos_xy1[2])[:2] # (x,y,r^2,r)
    final_pos = (int(final_pos_xy[0]),  int(final_pos_xy[1]))
    
    out_dict['move_dir'] = direction_vector(final_pos,(y,x))
    print(position_world,final_position_world[:3],final_postion_cam,position_cam)

    out_dict['move_dir_3d_gt'] = normalize_vector(position_world,final_position_world[:3])
    print(out_dict['move_dir'],out_dict['move_dir_3d_gt'],normalize_vector(position_cam[:3],final_postion_cam[:3]),np.linalg.inv(cam.get_metadata()['mat44'])[:3,:3]@out_dict['move_dir_3d_gt'])
    # exit()
    
    final_draw =   (y-10*out_dict['move_dir'][0],x-10*out_dict['move_dir'][1])
    draw.line([final_draw,(y,x)], fill="yellow", width=5)
    rgb_img.save(os.path.join(out_dir, 'move_dir_hint4.png'))
    # exit()
    

except ContactError:
    success = False
    mani_success = False
    # shutil.rmtree(out_dir)
    env.close()
    exit(1)
#print("1111")
rgb_final_pose, final_depth = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose_replay.png'))
# exit()
target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_world_xyz1_end = target_link_mat44 @ position_local_xyz1

touch_position_world_xyz_end = position_world_xyz1_end[:3]


if success:
    final_target_part_qpos = env.get_target_part_qpos()
    if primact_type == 'pushing':
        abs_motion = abs(final_target_part_qpos - start_target_part_qpos)
        j = replay_data['target_object_part_joint_id']
        tot_motion = replay_data['joint_angles_upper'][j] - replay_data['joint_angles_lower'][j] + 1e-8
        mani_success = (abs_motion > 0.01) or (abs_motion / tot_motion > 0.5)
        
    elif primact_type == 'pulling':
        abs_motion = abs(final_target_part_qpos - start_target_part_qpos)
        j = replay_data['target_object_part_joint_id']
        tot_motion = replay_data['joint_angles_upper'][j] - replay_data['joint_angles_lower'][j] + 1e-8
        mov_dir = np.array(touch_position_world_xyz_end, dtype=np.float32) - \
                        np.array(touch_position_world_xyz_start, dtype=np.float32)
        mov_dir /= np.linalg.norm(mov_dir)
        if move_dir:
            if move_dir =='up':
                intended_dir = -np.array(up, dtype=np.float32)
            else:
                intended_dir = -np.array(left, dtype=np.float32)
        mani_success = ((abs_motion > 0.01) or (abs_motion / tot_motion > 0.5) ) and (intended_dir @ mov_dir > 0.5)
        # print(move_dir,abs_motion,abs_motion / tot_motion,intended_dir @ mov_dir)
    print('-------------------------',success, mani_success)
    if not mani_success:
        shutil.rmtree(out_dir)
        env.close()
        exit(1)
    else:
        with open(os.path.join(out_dir,'result.json'), "r") as file:
            existing_data = json.load(file)
        existing_data.update(out_dict)
        with open(os.path.join(out_dir,'result.json'), "w") as file:
            json.dump(existing_data, file)
        env.close()
        exit(1)
        
else:
    shutil.rmtree(out_dir)
    env.close()
    exit(1)




