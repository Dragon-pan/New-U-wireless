from os.path import join,exists
import h5py
import json
import numpy as np
import natsort
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

def tensor_from_names(names, hfile):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    desc = [hfile[i]['global_descriptor'].__array__() for i in names]
    desc = torch.from_numpy(np.stack(desc, 0)).to(device).float()
    return desc

def load_destination(path):
    with open(path, 'r') as f:
        destinations = json.load(f)
    return destinations

def load_map(path):
    with open(path, 'r') as f:
        data = json.load(f)
        keyframe = data['keyframes']
        keyframe=dict(sorted(keyframe.items()))
        landmarks=data['landmarks']
        database_name,database_loc,registered_feature,keyframe_name,keyframe_location=[],[],[],[],[]
        T = np.array(data['T'])
        rot_base = np.arctan2(T[1, 0], T[0, 0])
        for k,v in keyframe.items():
            landmark_ids=v['lm_ids']
            landmarks_temp = []
            for i in landmark_ids:
                landmark = landmarks[str(i)]
                landmarks_temp.append([landmark['x'], landmark['y'], landmark['z']])
            registered_feature.append([keyframe[k]['kp_index'],landmarks_temp])
            database_name.append(k+'.png')
            database_loc.append(v['trans']+[v['rot']])
            dbname_list=k.split('_')
            dbyaw=dbname_list[-1]
            key_name=dbname_list[0]
            if dbyaw == '00':
                keyframe_name.append(key_name)
                keyframe_location.append(v['trans'])
        logger.info(f'Loaded {len(keyframe_name)} keyframes data')
    return database_name,database_loc,registered_feature,keyframe_name,keyframe_location,T,rot_base

def load_local_feature(path):
    hfile_local = h5py.File(path, 'r')
    logger.info(f'Loaded {len(hfile_local)} local descriptors')
    return hfile_local

def load_global_descriptor(path,kf_name):
    names = []
    hfile = h5py.File(path, 'r')
    hfile.visititems(
    lambda _, obj: names.append(obj.parent.name.strip('/'))
    if isinstance(obj, h5py.Dataset) and obj.parent.name.strip('/') in kf_name else None)
    keyframe_desc = tensor_from_names(names, hfile)
    logger.info(f'Loaded {len(keyframe_desc)} global descriptors')
    return keyframe_desc

def load_boundaires(path):
    with open(path, 'r') as f:
        data = json.load(f)
        lines = data['lines']
        add_lines = data['add_lines']
        for i in add_lines:
            lines.append(i)
        destinations = data['destination']
        anchor_name,anchor_location=[],[]
        for k, v in destinations.items():
            ll = k.split('-')
            anchor_name.append(v['id'])
            anchor_location.append([int(ll[0]), int(ll[1])])
        targets_num = len(destinations)
        for k, v in data['waypoints'].items():
            anchor_name.append(k)
            anchor_location.append(v)
        logger.info(f'Loaded {len(anchor_name)} anchor points and {len(lines)} boundaries')
    return anchor_name,anchor_location,lines

def load_graph(path):
    access_graph = np.load(path)
    return access_graph

def load_data(config):
    location_config=config['location']
    path = join(config['IO_root'], 'data',location_config['place'],location_config['building'],str(location_config['floor'])) ### data root path
    
    logger.info('============Loading map=============')
    paths={}
    paths['Destination']=join(config['IO_root'],'data','destination.json')
    paths['Map']=join(path, 'topo-map.json')
    paths['Local feature']=join(path, "feats-superpoint.h5")
    paths['Global descriptor']=join(path, "global_features.h5")
    paths['Boundaries']=join(path, 'boundaries.json')
    paths['Access graph path']=join(path, 'access_graph.npy')

    for key,path in paths.items():
        if not exists(path):
            logger.error(f"{key} file does not exist!")
            exit()  
    
    map_data={}

    """
    Load destination list
    """
    destinations=load_destination(paths['Destination'])
    map_data['destinations']=destinations

    """
    Load Topometric map
    """
    database_name,database_loc,registered_feature,keyframe_name,keyframe_location,T,rot_base=load_map(paths['Map'])
    map_data['database_name']=database_name
    map_data['database_loc']=database_loc
    map_data['registered_feature']=registered_feature
    map_data['keyframe_name']=keyframe_name
    map_data['keyframe_location']=np.array(keyframe_location)
    map_data['T']=T
    map_data['rot_base']=rot_base

    """
    Load local feature
    """
    hfile_local=load_local_feature(paths['Local feature'])
    map_data['local_descriptor']=hfile_local

    """
    Load global descriptor
    """
    keyframe_desc=load_global_descriptor(paths['Global descriptor'], set(database_name))
    map_data['global_descriptor']=keyframe_desc

    """
    Load boundaries
    """
    anchor_name,anchor_location,lines=load_boundaires(paths['Boundaries'])
    map_data['anchor_name']=anchor_name
    map_data['anchor_location']=anchor_location
    map_data['lines']=lines
    
    """
    Load shortest path
    """
    access_graph=load_graph(paths['Access graph path'])
    map_data['graph']=access_graph

    logger.info('==========Finished Loading==========')
    return map_data