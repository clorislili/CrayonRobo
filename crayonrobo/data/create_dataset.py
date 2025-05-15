import json
import os
import numpy as np
import random
import os

count = 0
folder_dir = '../data_collection/data/train_dataset' #the dir of collected data
folder_names = os.listdir(folder_dir)
random.shuffle(folder_names)
output_dir = './data/train_data_json' #dir to save json files for training
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cal_cat = dict()
for item in folder_names:
    
    cur_dir = os.path.join(folder_dir,str(item))

    cat = item.split('_')[1]
    if cat not in cal_cat.keys():
        cal_cat[cat] = 1
    else:
        if cal_cat[cat] > 600:
            continue
        else:
            cal_cat[cat] += 1
    
    if os.path.exists(os.path.join(cur_dir, 'result.json')):
        with open(os.path.join(cur_dir, 'result.json'), 'r') as fin:
            data_inf = json.load(fin)
            x,y = data_inf['pixel_locs']
            up_cam = data_inf['gripper_up_direction_camera']
            up_cam /= np.linalg.norm(up_cam)
            for_cam = data_inf['gripper_forward_direction_camera']
            for_cam /= np.linalg.norm(for_cam)
            left_cam = np.cross(up_cam, for_cam).tolist()
            left_cam /= np.linalg.norm(left_cam)
            
            up_world = data_inf['gripper_up_direction_world']
            up_world /= np.linalg.norm(up_world)
            for_world = data_inf['gripper_forward_direction_world']
            for_world /= np.linalg.norm(for_world)
            left_world = np.cross(up_world, for_world).tolist()
            left_world /= np.linalg.norm(left_world)
            try:
                up_vec = data_inf['up_dir_new']
                left_vec = data_inf['left_dir_new']
                move_dir = data_inf['move_dir']
                move_dir_3d_gt_1 = data_inf['move_dir_3d_gt']
                move_dir_3d_gt_1 /= np.linalg.norm(move_dir_3d_gt_1)
            except:
                continue
            
            move_dir_3d_gt_1_cam =  np.linalg.inv(data_inf['camera_metadata']['mat44'])[:3,:3] @ move_dir_3d_gt_1

            move_dir_3d_gt = []
            for i in move_dir_3d_gt_1_cam:
                move_dir_3d_gt.append(round(i,4))
            
            gt = f"The contact point at ({int(x)}, {int(y)}),  the gripper up 3D direction is {up_cam.tolist()}, the gripper left 3D direction is {left_cam.tolist()}."
            hint1 = 'Specify the contact point and orientation of pulling the object. There are some hints in the image: contact point is shown in blue dot. Specifically, the contact point is at ({},{})'.format(int(x),int(y))
            hint2 = 'Specify the contact point and orientation of pulling the object. There are some hints in the image: contact point is shown in blue dot, gripper up 2d direction is shown in red line, gripper left 2D direction is shown in green line. Specifically, The contact point is at ({},{}), the gripper up 2d direction is {}'.format(int(x),int(y),up_vec)
            hint3 = 'Specify the contact point and orientation of pulling the object. There are some hints in the image: contact point is shown in blue dot, gripper up 2d direction is shown in red line, gripper left 2D direction is shown in green line. Specifically, The contact point is at ({},{}), the gripper up 2d direction is {}, the gripper left 2d direction is {}'.format(int(x),int(y),up_vec, left_vec)
            hint4 = 'Specify the contact point and orientation of pulling the object. There are some hints in the image: contact point is shown in blue dot, gripper up 2d direction is shown in red line, gripper left 2D direction is shown in green line. Specifically, The contact point is at ({},{}), the gripper left 2d direction is {}'.format(int(x),int(y),left_vec)
            hint5 = 'Specify the contact point and orientation of pulling the object. There are some hints in the image: contact point is shown in blue dot, gripper up 2d direction is shown in red line, gripper left 2D direction is shown in green line, the 2D moving direction is shown in yellow line. Specifically, The contact point is at ({},{}), the gripper up 2d direction is {}, the gripper left 2d direction is {}, the moving 2D direction is {}.'.format(int(x),int(y),up_vec, left_vec, move_dir)
            imagehint = 'Specify the contact point and orientation of pulling the object. There are some hints in the image: contact point is shown in blue dot, gripper up 2d direction is shown in red line, gripper left 2D direction is shown in green line.'
            gt2 = f"The contact point at ({int(x)}, {int(y)}),  the gripper up 3D direction is {up_cam.tolist()}, the gripper left 3D direction is {left_cam.tolist()}, the moving 3D direction is {move_dir_3d_gt}."
            
            data = {
                "image": os.path.join(cur_dir, 'rgb.png'),
                "image_hint4": os.path.join(cur_dir, 'move_dir_hint4.png'),
                "conversations": [
                    {
                        "prompt": "Specify the contact point and orientation of pulling the object."
                    },
                    {
                        "gt": gt
                        #f"The contact point at ({int(x)}, {int(y)}),  the gripper direction is {action_direction_cam.tolist()}, the gripper forward direction is {forward_cam.tolist()}."


                    }
                ],
                "instruction": "Specify the contact point and orientation of pulling the object.",
                "input": os.path.join(cur_dir, 'rgb.png'),
                "output": gt,
                'hint1': hint1,
                'hint2': hint2,
                'hint3': hint3,
                'hint4':hint4,
                'prompt4image': imagehint,
                'gt2': gt2,
                'hint5': hint5
                

            }
            
            json_data = json.dumps(data, indent=4)

            
            with open(os.path.join(output_dir,'{}.json'.format(item)), "w") as file:
                file.write(json_data)


print('The training samples for each category:{}. \nThe number of training categories is: {}'.format(cal_cat,len(cal_cat.keys())))