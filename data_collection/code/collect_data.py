
import os
import sys
import shutil
import numpy as np
from PIL import Image, ImageDraw
from utils import get_global_position_from_camera, save_h5
import cv2
import json
import gc
from argparse import ArgumentParser
import stat
import subprocess
from sapien.core import Pose
from env import Env, ContactError
from camera import Camera
from robots.panda_robot import Robot
import math
import time
file_path = os.path.abspath(__file__)

parser = ArgumentParser()
parser.add_argument('shape_id', type=str)
parser.add_argument('category', type=str)
parser.add_argument('cnt_id', type=int)
parser.add_argument('primact_type', type=str)
parser.add_argument('--out_dir', default="results", type=str)
parser.add_argument('--trial_id', type=int, default=0, help='trial id')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--no_gui', action='store_true', default=False, help='no_gui [default: False]')
args = parser.parse_args()
def handle_remove_readonly(func, path, exc_info):
    """处理因权限导致的删除失败"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def normalize_vector(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    vx = x2 - x1
    vy = y2 - y1
    vz = z2 - z1
    vector_length = math.sqrt(vx**2 + vy**2 + vz**2)

    normalized_vx = vx / vector_length
    normalized_vy = vy / vector_length
    normalized_vz = vz / vector_length

    return (normalized_vx, normalized_vy, normalized_vz)
def direction_vector(loc1, loc2):
    x1, y1 = loc1
    x2, y2 = loc2
    dx = x2 - x1
    dy = y2 - y1
    
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    if magnitude != 0:
        direction_vector = (dx / magnitude, dy / magnitude)
    else:
        direction_vector = (0, 0)  # If the points are the same, return zero vector
        
    return direction_vector

def distance_between_points(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance


shape_id = args.shape_id
trial_id = args.trial_id
primact_type = args.primact_type


if args.no_gui:
    out_dir = os.path.join(args.out_dir, '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))
else:
    out_dir = os.path.join('results', '%s_%s_%d_%s_%d' % (shape_id, args.category, args.cnt_id, primact_type, trial_id))
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
print("out_dir:", out_dir)

os.makedirs(out_dir)
flog = open(os.path.join(out_dir, 'log.txt'), 'w')
out_info = dict()

if args.random_seed is not None:
    np.random.seed(args.random_seed)
    out_info['random_seed'] = args.random_seed

# setup env
env = Env(flog=flog, show_gui=(not args.no_gui))

# setup camera
cam = Camera(env, image_size=996,random_position=True)
out_info['camera_metadata'] = cam.get_metadata_json()
if not args.no_gui:
    env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi+cam.theta, -cam.phi)

# load shape
object_urdf_fn = '../assets/%s/mobility.urdf' % shape_id

flog.write('object_urdf_fn: %s\n' % object_urdf_fn)
object_material = env.get_material(4, 4, 0.01)
if primact_type == 'pulling':
    state = 'random-closed-middle'
    if np.random.random() < 0.2:
        state = 'closed'
else:
    state = 'random-closed-middle'
    if np.random.random() < 0.5:
        state = 'random-middle'
flog.write('Object State: %s\n' % state)
out_info['object_state'] = state
scale = np.random.uniform(low=0.7, high=1.2)
out_info['scale'] = scale

joint_angles = env.load_object(object_urdf_fn, object_material, scale,state=state)

out_info['joint_angles'] = joint_angles
out_info['joint_angles_lower'] = env.joint_angles_lower
out_info['joint_angles_upper'] = env.joint_angles_upper
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
    flog.write('Object Not Still!\n')
    # print("Object is not placed still ---->  Delete!!!!")
    flog.close()
    env.close()
    shutil.rmtree(out_dir)
    exit(1)

### use the GT vision
rgb, depth = cam.get_observation()
depth_before = depth
Image.fromarray((rgb*255).astype(np.uint8)).save(os.path.join(out_dir, 'rgb.png'))
img = Image.fromarray((rgb*255).astype(np.uint8))
draw = ImageDraw.Draw(img)

cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA_before = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])

gt_nor = cam.get_normal_map()
Image.fromarray(((gt_nor+1)/2*255).astype(np.uint8)).save(os.path.join(out_dir, 'gt_nor.png'))

object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
Image.fromarray((gt_movable_link_mask>0).astype(np.uint8)*255).save(os.path.join(out_dir, 'interaction_mask.png'))

# sample a pixel to interact
xs, ys = np.where(gt_movable_link_mask>0)
if len(xs) == 0:
    flog.write('No Movable Pixel! Quit!\n')
    flog.close()
    env.close()
    exit(1)
idx = np.random.randint(len(xs))
x, y = xs[idx], ys[idx]

# color the contact point
draw.point((y,x),'red')
img.save(os.path.join(out_dir, 'contact_point.png'))

out_info['pixel_locs'] = [int(x), int(y)]
env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y]-1])
out_info['target_object_part_actor_id'] = env.target_object_part_actor_id
out_info['target_object_part_joint_id'] = env.target_object_part_joint_id
part_movable_link_mask = cam.get_movable_link_mask([object_link_ids[gt_movable_link_mask[x, y]-1]])

# get pixel 3D pulling direction (cam/world)
direction_cam = gt_nor[x, y, :3]
direction_cam /= np.linalg.norm(direction_cam)

out_info['direction_camera'] = direction_cam.tolist()
flog.write('Direction Camera: %f %f %f\n' % (direction_cam[0], direction_cam[1], direction_cam[2]))
direction_world = cam.get_metadata()['mat44'][:3, :3] @ direction_cam
out_info['direction_world'] = direction_world.tolist()
flog.write('Direction World: %f %f %f\n' % (direction_world[0], direction_world[1], direction_world[2]))
flog.write('mat44: %s\n' % str(cam.get_metadata()['mat44']))

# sample a random direction in the hemisphere (cam/world)
action_direction_cam = -gt_nor[x, y, :3]+np.random.uniform(-0.05, 0.05, size=(1, 3))[0]
action_direction_cam /= np.linalg.norm(action_direction_cam)

out_info['gripper_direction_camera'] = action_direction_cam.tolist()
action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ action_direction_cam
out_info['gripper_direction_world'] = action_direction_world.tolist()

# get pixel 3D position (cam/world)
position_cam = cam_XYZA_before[x, y, :3]
out_info['position_cam'] = position_cam.tolist()
position_cam_xyz1 = np.ones((4), dtype=np.float32)
position_cam_xyz1[:3] = position_cam
position_world_xyz1 = cam.get_metadata()['mat44'] @ position_cam_xyz1
position_world = position_world_xyz1[:3]
out_info['position_world'] = position_world.tolist()

# compute final pose
up = np.array(action_direction_world, dtype=np.float32)#向上
up /= np.linalg.norm(up)
# camera_matrix: camera homogeneous coordinate->image homogeneous coordinate
up_point2 = position_cam - 0.13*action_direction_cam
up_point2 = [-up_point2[1],-up_point2[2],up_point2[0]]
up_xy1 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],up_point2)

up_xy = (up_xy1/up_xy1[2])[:2] # (x,y,r^2,r)
up_2 = (int(up_xy[0]),  int(up_xy[1]))
draw.line([(y,x),up_2], fill="red", width=5)
up_vec = direction_vector((up_xy[1],up_xy[0]),(x,y))
out_info["up_vec"] = up_vec

forward = np.random.randn(3).astype(np.float32)
while abs(up @ forward) > 0.99:
    forward = np.random.randn(3).astype(np.float32)
forward /= np.linalg.norm(forward)
left = np.cross(up, forward)
left /= np.linalg.norm(left)
forward = np.cross(left, up)
forward /= np.linalg.norm(forward)

left_cam = np.linalg.inv(cam.get_metadata()['mat44']) @ np.append(left,0)
left_point1_cam = position_cam - 0.05*left_cam[:3]
left_point2_cam = position_cam + 0.05*left_cam[:3]
left_point1_cam = [-left_point1_cam[1],-left_point1_cam[2],left_point1_cam[0]]
left_point2_cam = [-left_point2_cam[1],-left_point2_cam[2],left_point2_cam[0]]
left_xy1 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],left_point1_cam)
left_xy1 = (left_xy1/left_xy1[2])[:2] 
left_xy2 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],left_point2_cam)
left_xy2 = (left_xy2/left_xy2[2])[:2] 
left_vec = direction_vector((left_xy2[1],left_xy2[0]),(x,y))
left_2 = (int(left_xy2[0]),  int(left_xy2[1]))
draw.line([(y,x),left_2], fill="green", width=5)
radius = 3
pixel_color = "blue"

# Draw a filled ellipse (pixel)
draw.ellipse((y - radius, x - radius, 
              y + radius, x + radius), fill=pixel_color)

img.save(os.path.join(out_dir, 'pull_force.png'))
# save in json file
out_info["left_vec"] = left_vec

out_info['gripper_forward_direction_world'] = forward.tolist()
forward_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ forward
out_info['gripper_forward_direction_camera'] = forward_cam.tolist()

up /= np.linalg.norm(up)
out_info['gripper_up_direction_world'] = up.tolist()
up_cam = np.linalg.inv(cam.get_metadata()['mat44'][:3, :3]) @ up
out_info['gripper_up_direction_camera'] = up_cam.tolist()

rotmat = np.eye(4).astype(np.float32)
rotmat[:3, 0] = forward
rotmat[:3, 1] = left
rotmat[:3, 2] = up

final_dist = 0.11
if primact_type == 'pushing-left' or primact_type == 'pushing-up':
    final_dist = 0.11

mid_rotmat = np.array(rotmat, dtype=np.float32)
mid_rotmat[:3, 3] = position_world - action_direction_world * final_dist
mid_pose = Pose().from_transformation_matrix(mid_rotmat)
out_info['mid_rotmat_world'] = mid_rotmat.tolist()

start_rotmat = np.array(rotmat, dtype=np.float32)
start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
start_pose = Pose().from_transformation_matrix(start_rotmat)
out_info['start_rotmat_world'] = start_rotmat.tolist()


pull_rotmat = np.array(rotmat, dtype=np.float32)
pull_rotmat[:3, 3] = position_world - action_direction_world * (final_dist+0.005)
pull_pose = Pose().from_transformation_matrix(pull_rotmat)



pull_rotmat_left = np.array(rotmat, dtype=np.float32)
pull_rotmat_left[:3, 3] = position_world - left * (final_dist+0.005)
pull_pose_left = Pose().from_transformation_matrix(pull_rotmat_left)



### viz the EE gripper position
# setup robot
robot_urdf_fn = './robots/panda_gripper.urdf'
robot_material = env.get_material(4, 4, 0.01)
robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))


# move back
robot.open_gripper()
robot.robot.set_root_pose(start_pose)
env.render()

rgb_final_pose, _ = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_start_pose.png'))

env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in primact_type)

out_info['start_target_part_qpos'] = env.get_target_part_qpos()

target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

success = True

try:
    robot.move_to_target_pose(mid_rotmat, 2000)
    robot.close_gripper()
    robot.wait_n_steps(2000)
    rgb_final_pose, _ = cam.get_observation()
    Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_mid_pose.png'))

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
        out_info['target_rotmat_world'] = final_rotmat.tolist()
    

except ContactError:
    print('Contact error------->Delete!!!!!')
    
    success = False
    mani_success = False
    
    flog.close()
    env.close()
    shutil.rmtree(out_dir)
    exit(1)

rgb_final_pose, final_depth = cam.get_observation()
Image.fromarray((rgb_final_pose*255).astype(np.uint8)).save(os.path.join(out_dir, 'viz_target_pose.png'))

target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
position_world_xyz1_end = target_link_mat44 @ position_local_xyz1
flog.write('touch_position_world_xyz_start: %s\n' % str(position_world_xyz1))
flog.write('touch_position_world_xyz_end: %s\n' % str(position_world_xyz1_end))
out_info['touch_position_world_xyz_start'] = position_world_xyz1[:3].tolist()
out_info['touch_position_world_xyz_end'] = position_world_xyz1_end[:3].tolist()
out_info['gt'] = f"The contact point at ({int(x)}, {int(y)}),  the gripper direction is {action_direction_cam.tolist()}, the gripper forward direction is {forward_cam.tolist()}."
final_position_world = target_link_mat44 @ position_local_xyz1
final_postion_cam =  np.linalg.inv(cam.get_metadata()['mat44']) @ final_position_world
final_postion_cam2 = [-final_postion_cam[1],-final_postion_cam[2],final_postion_cam[0]]
final_pos_xy1 = np.dot((cam.get_metadata())["camera_matrix"][:3,:3],final_postion_cam2)
final_pos_xy = (final_pos_xy1/final_pos_xy1[2])[:2] # (x,y,r^2,r)
final_pos = (int(final_pos_xy[0]),  int(final_pos_xy[1]))
out_info['up_dir_new'] = direction_vector((y,x),up_2)
out_info['left_dir_new'] = direction_vector((y,x),left_2)
out_info['move_dir'] = direction_vector(final_pos,(y,x))
out_info['move_dir_3d_gt'] = normalize_vector(position_world,final_position_world[:3])
final_draw =   (y-10*out_info['move_dir'][0],x-10*out_info['move_dir'][1])
draw.line([final_draw,(y,x)], fill="yellow", width=5)
img.save(os.path.join(out_dir, 'move_dir_hint4.png'))

if success:
    out_info['result'] = 'VALID'
    out_info['final_target_part_qpos'] = env.get_target_part_qpos()
    if primact_type == 'pushing':
        abs_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])
        j = out_info['target_object_part_joint_id']
        tot_motion = out_info['joint_angles_upper'][j] - out_info['joint_angles_lower'][j] + 1e-8
        mani_success = (abs_motion > 0.01) or (abs_motion / tot_motion > 0.5)
        out_info['mani_succ'] = str(mani_success)
    elif primact_type == 'pulling':
        abs_motion = abs(out_info['final_target_part_qpos'] - out_info['start_target_part_qpos'])
        j = out_info['target_object_part_joint_id']
        tot_motion = out_info['joint_angles_upper'][j] - out_info['joint_angles_lower'][j] + 1e-8
        mov_dir = np.array(out_info['touch_position_world_xyz_end'], dtype=np.float32) - \
                        np.array(out_info['touch_position_world_xyz_start'], dtype=np.float32)
        norm = np.linalg.norm(mov_dir)
        mov_dir = mov_dir / norm if norm > 1e-6 else np.zeros_like(mov_dir)
        if move_dir:
            if move_dir =='up':
                intended_dir = -np.array(out_info['gripper_direction_world'], dtype=np.float32)
            else:
                intended_dir = -np.array(left, dtype=np.float32)
        mani_success = ((abs_motion > 0.01) or (abs_motion / tot_motion > 0.5) ) and (intended_dir @ mov_dir > 0.5)
        # print('-----result',abs_motion / tot_motion,intended_dir @ mov_dir)
        out_info['mani_succ'] = str(mani_success)
    
    if mani_success:
        with open(os.path.join(out_dir, 'result.json'), 'w') as fout:
            json.dump(out_info, fout)
    else:
        flog.close()
        env.close()
        shutil.rmtree(out_dir)
        exit(1)
else:
    
    
    flog.close()
    env.close()
    shutil.rmtree(out_dir)
    exit(1)


print('---------------collected one success episode')

flog.close()
env.close()


