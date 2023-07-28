from third_party.local_feature.SuperPoint_SuperGlue.base_model import dynamic_load
from third_party.local_feature.SuperPoint_SuperGlue import extractors,matchers
import numpy as np
import torch
import cv2

class Superpoint():
    def __init__(self,device,conf):
        Model_sp = dynamic_load(extractors, conf['detector_name'])
        self.local_feature_extractor=Model_sp({'name':conf['detector_name'],'nms_radius':conf['nms_radius'],'max_keypoints':conf['max_keypoints']}).eval().to(device)
        self.device=device

    def prepare_data(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.).unsqueeze(0)
        return data

    def extract_local_features(self, image0):
        data0 = self.prepare_data(image0)
        pred0 = self.local_feature_extractor(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if 'keypoints' in pred0:
            pred0['keypoints'] = (pred0['keypoints'] + .5) - .5
        pred0.update({'image_size': np.array([image0.shape[0], image0.shape[1]])})
        return pred0

class Superpoint_class():
    def __init__(self,device,**config):
        self.local_feature_extractor = loadModel(device,config['vpr']['local_feature']['path'])
        self.device=device

    def extract_local_features(self,image):
        params = {
            'out_num_points': 500,
            'patch_size': 5,
            'device': self.device,
            'nms_dist': 4,
            'conf_thresh': 0.015
        }
        image=image.to(self.device)
        out=self.local_feature_extractor(image)
        return out

class Local_extractor():
    def __init__(self,configs):
        self.configs=configs
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

    def superglue(self,conf):
        Model_sg = dynamic_load(matchers, conf['matcher_name'])
        return Model_sg({'name':conf['matcher_name'],'weights':conf['weights'],'sinkhorn_iterations':conf['sinkhorn_iterations']}).eval()

    def extractor(self):
        for name,content in self.configs.items():
            if name=='superpoint+superglue':
                superpoint=Superpoint(self.device,self.configs['superpoint+superglue'])
                return superpoint.extract_local_features

            elif name=='sift':
                pass

            elif name=='surf':
                pass

    def matcher(self):
        for name,content in self.configs.items():
            if name == 'superpoint+superglue':
                return self.superglue(self.configs['superpoint+superglue'])
            elif name == 'sift':
                pass
            elif name == 'surf':
                pass