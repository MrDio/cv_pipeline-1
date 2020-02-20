
import numpy as np
import json
import sys
sys.path.append('.')
sys.path.append('./gqcnn')
sys.path.append('./sd_maskrcnn')
sys.path.append('./sd_maskrcnn/maskrcnn')
from dexnet.network import DexnetLoader
from dexnet.maskNet import MaskLoader
from cv_pipeline.srv import gqcnnpj, gqcnnsuction, fcgqcnnsuction, fcgqcnnpj, maskrcnn
import tensorflow as tf
import cv_bridge
import rospy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

bridge = cv_bridge.CvBridge()


def _handle_gqcnn(req, func):
    rgb = bridge.imgmsg_to_cv2(req.rgb, "rgb8")
    depth = bridge.imgmsg_to_cv2(req.depth, "32FC1")
    intrinsics = json.loads(req.intrinsics)
    res = func(rgb, depth, intrinsics)
    return json.dumps(res)
    

def _handle_fcgqcnn(req, func):
    rgb = bridge.imgmsg_to_cv2(req.rgb, "rgb8")
    depth = bridge.imgmsg_to_cv2(req.depth, "32FC1")
    mask = bridge.imgmsg_to_cv2(req.mask, "rgb8")
    intrinsics = json.loads(req.intrinsics)
    res = func(rgb, depth, mask, intrinsics)
    return json.dumps(res)

def _handle_mask(req, func):
    depth = bridge.imgmsg_to_cv2(req.depth, "32FC1")
    res = func(depth)
    masks = []
    for mask in res["masks"]:
        mask_img = bridge.cv2_to_imgmsg(np.array(mask), "rgb8")
        masks.append(mask_img)
    print(res["masks"])
    return masks

class PipelineService():
    
    def __init__(self):
        self.mask_net = MaskLoader()
        self.gqcnnpj_net = DexnetLoader('cfg/gqcnn_pj_tuned.yaml')
        self.gqcnnsuction_net = DexnetLoader('cfg/gqcnn_suction.yaml')
        self.fcgqcnnpj_net = DexnetLoader('cfg/fcgqcnn_pj.yaml')
        self.fcgqcnnsuction_net = DexnetLoader('cfg/fc_gqcnn_suction.yaml')
    
    def start(self):

        rospy.Service("gqcnnpj", gqcnnpj, lambda req: _handle_gqcnn(req, lambda img, d, intrinsics: self.gqcnnpj_net.predict(self.gqcnnpj_net.rgbd2state(img, d, intrinsics))))
        rospy.Service("gqcnnsuction", gqcnnsuction, lambda req: _handle_gqcnn(req, lambda img, d, intrinsics: self.gqcnnsuction_net.predict(self.gqcnnsuction_net.rgbd2state(img, d, intrinsics))))
        rospy.Service("fcgqcnnpj", gqcnnpj, lambda req: _handle_fcgqcnn(req, lambda img, d, mask, intrinsics: self.fcgqcnnpj_net.predict(self.fcgqcnnpj_net.rgbd2state(img, d, mask, intrinsics))))
        rospy.Service("fcgqcnnsuction", gqcnnsuction, lambda req: _handle_fcgqcnn(req, lambda img, d, mask ,intrinsics: self.fcgqcnnsuction_net.predict(self.fcgqcnnsuction_net.rgbd2state(img, d, mask, intrinsics))))
        rospy.Service("maskrcnn", maskrcnn, lambda req: _handle_mask(req, self.mask_net.predict))
        rospy.spin()



if __name__ == '__main__':
    rospy.init_node('cv_pipeline')
    pipe_service = PipelineService()
    pipe_service.start()
