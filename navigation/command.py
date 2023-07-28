import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

def actions(current_pose,path_list,scale):
    xc, yc, an = current_pose
    action_list = []
    for p in path_list:
        xn,yn=p
        distance = np.linalg.norm([xn - xc, yn - yc])
        rot = np.arctan2(xn - xc, yn - yc)/ np.pi * 180+180
        rot_ang = (an - rot) % 360
        rot_clock = (round(rot_ang.squeeze().tolist() / 15) / 2) % 12
        if rot_clock < 1:
            rot_clock += 12
        action_list.append([rot_clock, distance*scale])
        xc, yc ,an= xn, yn,rot
    return action_list

def command_type0(current_pose,path_list,scale):
    if current_pose:
        if len(path_list) > 0:
            action_list=actions(current_pose,path_list,scale)
            message = ''
            for i, ac in enumerate(action_list):
                rot_clock, distance = ac
                if rot_clock-int(rot_clock)==0.5:
                    message += 'Please walk %.1f meters along %d point 5 clock' % (
                        distance, int(rot_clock))
                else:
                    message += 'Please walk %.1f meters along %d clock' % (
                        distance, int(rot_clock))
                if i < len(action_list) - 1:
                    message += '. Then '
                else:
                    message += '. And you will arrive the destination.\n'
            return message
        else:
            return "There's no path to the destination"
    else:
        return "Cannot localize"

def command_type1(current_pose,path_list,scale):
    if current_pose:
        if len(path_list) > 0:
            action_list=actions(current_pose,path_list,scale)
            message = ''
            rot_clock,next_distance=action_list[0]
            next_station='your destination' if len(action_list)==1 else 'anchor points'
            if next_distance<5:
                if rot_clock-int(rot_clock)==0.5:
                    message += 'Please walk %.1f meters along %d point 5 clock' % (
                        next_distance, int(rot_clock))
                else:
                    message += 'Please walk %.1f meters along %d clock' % (
                        next_distance, int(rot_clock))
                message +=' and you will reach '+next_station
            return message
        else:
            return "There's no path to the destination"
    else:
        return "Cannot localize"