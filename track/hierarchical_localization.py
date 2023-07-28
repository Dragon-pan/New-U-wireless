import logging
from feature.global_extractor import Global_Extractors
from feature.local_extractor import Local_extractor
from feature.local_matcher import Local_matcher
from third_party.torchSIFT.src.torchsift.ransac.ransac import ransac
from third_party.torchSIFT.src.torchsift.ransac.matcher import match
from .implicit_distortion_model import coarse_pose,pose_multi_refine
import torch
import cv2
import numpy as np

class Hloc():
    device='cuda' if torch.cuda.is_available() else "cpu"
    def __init__(self, root, map_data, config):
        self.config=config['hloc']
        self.thre=self.config['ransac_thre']
        self.match_type=self.config['match_type']
        self.rot_base=map_data['rot_base']
        self.T=map_data['T']
        self.db_desc=map_data['global_descriptor']
        self.keyframe_name=map_data['keyframe_name']
        feature_configs=config['feature']
        global_extractor = Global_Extractors(root,feature_configs['global'])
        self.global_extractor=global_extractor.get()
        local_feature = Local_extractor(feature_configs['local'])
        self.local_feature_extractor = local_feature.extractor()
        self.local_feature_matcher=Local_matcher(map_data['database_name'],map_data['local_descriptor'],map_data['registered_feature'],**feature_configs)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        self.list_2d, self.list_3d, self.initial_poses, self.pps = [], [], [], []

    def global_retrieval(self,image):
        """
        Global Images Retrieval:
            Find the topk closest database images of input query image
        """
        self.query_desc = self.global_extractor.feature(image)
        sim = torch.einsum('id,jd->ij', self.query_desc, self.db_desc)
        topk = torch.topk(sim, self.config['retrieval_num'], dim=1).indices.cpu().numpy()
        return topk

    def feature_matching(self,image,topk):
        """
        Local Feature Matching:
            Match the local features between query image and retrieved database images
        """
        feats0 = self.local_feature_extractor(image)
        pts0_list,pts1_list,lms_list=[],[],[]
        max_len=0
        self.retrived_image_index=[]
        for i in topk[0]:
            pts0,pts1,lms=self.local_feature_matcher.superglue(i, feats0)
            feat_inliner_size=pts0.shape[0]
            if feat_inliner_size>self.thre:
                pts0_list.append(pts0)
                pts1_list.append(pts1)
                lms_list.append(lms)
                self.retrived_image_index.append(i)
                if feat_inliner_size>max_len:
                    max_len=feat_inliner_size
        self.retrived_image_index=torch.tensor(self.retrived_image_index).to(self.device)
        del self.query_desc, feats0
        return pts0_list,pts1_list,lms_list,max_len

    def geometric_verification(self,pts0_list,pts1_list,lms_list,max_len):
        """
        Geometric verification:
            Apply geometric verification between query and database images
        """
        batch_size=len(pts0_list)
        pts0=torch.empty((batch_size,max_len,2),dtype=float)
        pts1=torch.empty((batch_size,max_len,2),dtype=float)
        lms=torch.empty((batch_size,max_len,3),dtype=float)
        mask=torch.zeros((batch_size,max_len,max_len))
        for i in range(batch_size):
            inliner_size=len(pts0_list[i])
            pts0[i,:inliner_size,:]=torch.from_numpy(pts0_list[i])
            pts1[i,:inliner_size,:]=torch.from_numpy(pts1_list[i])
            lms[i,:inliner_size,:]=torch.from_numpy(lms_list[i])
            mask[i,:inliner_size,:inliner_size]=torch.ones((inliner_size,inliner_size))
        pts0,pts1,lms,mask=pts0.to(self.device),pts1.to(self.device),lms.to(self.device),mask.to(self.device)
        try:
            _,inliners,_=ransac(pts0,pts1,mask)
        except:
            return torch.tensor([]),None

        diag_masks = torch.diagonal(inliners, dim1=-2, dim2=-1)

        # Calculate sizes for each item in the batch
        sizes = diag_masks.sum(-1)

        # Filtering based on the threshold
        valid_indices = sizes > self.thre

        # Apply thresholding
        diag_masks = diag_masks[valid_indices]
        pts0 = pts0[valid_indices]
        lms = lms[valid_indices]
        self.retrived_image_index=self.retrived_image_index[valid_indices]

        # Masking pts0, pts1, and lms
        masked_pts0 = [pts0[i][diag_masks[i]] for i in range(pts0.size(0))]
        masked_lms = [lms[i][diag_masks[i]] for i in range(lms.size(0))]
        
        del pts0,pts1,lms,mask
        torch.cuda.empty_cache()
        if len(masked_pts0)>0:
            return torch.cat(masked_pts0),torch.cat(masked_lms)
        else:
            return torch.tensor([]),torch.tensor([])

    def nvs_ot(self,image,topk):
        feats0 = self.local_feature_extractor(image)
        fts0_list,desc0_list,fts1_list,desc1_list=[],[],[],[]
        max_len=feats0['keypoints'].size(0)
        for i in topk[0]:
            feat1=self.local_feature_matcher.hfile_local[self.local_feature_matcher.db_name[i]]
            fts0_list.append(feats0['keypoints'])
            desc0_list.append(torch.transpose(feat0['descriptors'],0,1))
            fts1_list.append(feat1['keypoints'])
            desc1_list.append(torch.transpose(feat1['descriptors'],0,1))
            size=feat1['keypoints'].size(0)
            if max_len<size:
                max_len=size
        fts0=torch.cat(fts0_list)
        desc0=torch.cat(desc0_list)
        print(fts0.size)

        

    def pnp(self,image,feature2D,landmark3D):
        """
        Start Perspective-n-points:
            Estimate the current location using implicit distortion model
        """
        if feature2D.size()[0]>0:
            height, width, _ = image.shape
            feature2D, landmark3D=feature2D.cpu().numpy(),landmark3D.cpu().numpy()
            out, p2d_inlier, p3d_inlier = coarse_pose(feature2D, landmark3D, np.array([width / 2, height / 2]))
            self.list_2d.append(p2d_inlier)
            self.list_3d.append(p3d_inlier)
            self.initial_poses.append(out['pose'])
            self.pps.append(out['pp'])
            if len(self.list_2d) > self.config['implicit_num']:
                self.list_2d.pop(0)
                self.list_3d.pop(0)
                self.initial_poses.pop(0)
                self.pps.pop(0)
            pose = pose_multi_refine(self.list_2d, self.list_3d, self.initial_poses, self.pps,self.rot_base,self.T)
        else:
            pose =None
            self.logger.warning("!!!Cannot localize at this point, please take some steps or turn around!!!")
        return pose

    def get_location(self, image):
        self.logger.info("Start image retrieval")
        topk=self.global_retrieval(image)

        if self.match_type=='superglue':
            self.logger.info("Matching local feature")
            pts0_list,pts1_list,lms_list,max_len=self.feature_matching(image,topk)

            self.logger.info("Start geometric verification")
            feature2D,landmark3D=self.geometric_verification(pts0_list, pts1_list, lms_list,max_len)
            
        elif self.match_type=='nvs':
            self.logger.info("Doing NVS+OT")
            feature2D,landmark3D=self.nvs_ot(image,topk)

        self.logger.info("Estimate the camera pose using PnP algorithm")
        pose=self.pnp(image,feature2D,landmark3D)

        return pose
        