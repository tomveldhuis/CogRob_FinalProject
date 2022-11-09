import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
import torch.utils.data
from PIL import Image
from datetime import datetime
import torch
from network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.visualisation.plot import plot_results
from network.utils.dataset_processing.grasp import detect_grasps
from skimage.filters import gaussian
from trained_models.Mask_RCNN.MRCNN import segmentImage
import os
import cv2

class GraspGenerator:
    IMG_WIDTH = 224
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius, fig, IMG_WIDTH=224, network='GR_ConvNet', device='cpu'):
        if (device=='cpu'):
            self.net = torch.load(net_path, map_location=device)
            self.device = get_device(force_cpu=True)
        else:
            #self.net = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(1))
            #self.device = get_device()
            print ("GPU is not supported yet! :( -- continuing experiment on CPU!" )
            self.net = torch.load(net_path, map_location='cpu')
            self.device = get_device(force_cpu=True)

        
        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius
        
        self.fig = fig
        self.network = network

        self.PIX_CONVERSION = 277 * IMG_WIDTH/224

        self.IMG_WIDTH = IMG_WIDTH

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(-img_center/self.PIX_CONVERSION,
                                                    img_center/self.PIX_CONVERSION,
                                                    0,
                                                    self.IMG_ROTATION)
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION)

    def get_transform_matrix(self, x, y, z, rot):
        return np.array([
                        [np.cos(rot),   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot),    0,  y],
                        [0,             0,              1,  z],
                        [0,             0,              0,  1]
                        ])

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = np.clip(x_p-self.depth_r, 0, self.IMG_WIDTH)
        x_max = np.clip(x_p+self.depth_r, 0, self.IMG_WIDTH)
        y_min = np.clip(y_p-self.depth_r, 0, self.IMG_WIDTH)
        y_max = np.clip(y_p+self.depth_r, 0, self.IMG_WIDTH)
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p /= self.PIX_CONVERSION
        y_p /= self.PIX_CONVERSION
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.IMG_ROTATION)
        if roll < -np.pi / 2:
            roll += np.pi

        # Covert pixel width to gripper width
        opening_length = (grasp.length / int(self.MAX_GRASP *
                          self.PIX_CONVERSION)) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        # return x, y, z, roll, opening length gripper
        return robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], roll, opening_length, obj_height

    def post_process_output(self, q_img, cos_img, sin_img, width_img, pixels_max_grasp):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * pixels_max_grasp

        q_img = gaussian(q_img, 1.0, preserve_range=True)
        ang_img = gaussian(ang_img, 1.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img

    def maskRCNN(self, desired_object, method, rgb, img_size, q_img, show_output):
        """ MASK R CNN HERE """
        #cv2.imwrite("originalRGB.png", rgb)
        print('Mask R CNNing')     
        confidence_threshold = 0.9
        objectBox, objectMask = segmentImage(rgb, desired_object, confidence_threshold, save_output_image=show_output) 
        
        ## resize the image for the grasp network
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation = cv2.INTER_AREA)
        print("objectBox", objectBox)
        ## resize the object box
        if objectBox != False:
            if method == 'boundingBox':
                # reshape box to fit the grasping network
                scale_factor = self.IMG_WIDTH / 500
                objectBox[0] = (int(objectBox[0][0] * scale_factor), int(objectBox[0][1] * scale_factor))
                objectBox[1] = (int(objectBox[1][0] * scale_factor), int(objectBox[1][1] * scale_factor))

                # unpack box edges
                print("MASKING Q-IMAGE")
                left_bound, top_bound, = objectBox[0]
                right_bound, bottom_bound = objectBox[1]
            
                q_img[  :top_bound,      :] = 0         # everything above the box
                q_img[bottom_bound:,      :] = 0         # everything below the box
                q_img[:,             :left_bound] = 0    # left of the box
                q_img[:,             right_bound:] = 0   # right of the box
                return True, q_img

            elif method == 'objectMask':
                print("object mask outline not implemented yet")
                return True, q_img

        else:
            print("object not found!")
            return False, q_img



    def predict(self, rgb, img_size, depth, n_grasps=1, show_output=False, desired_object='mustard_bottle', maskingMethod='boundingBox'):
        max_val = np.max(depth)
        depth = depth * (255 / max_val)
        depth = np.clip((depth - depth.mean())/175, -1, 1)
        
        # hold on to a copy of the 500x500 rgb image for maskRCNN later
        # and resize the 'original' rgb image for the grasping network
        maskRCNN_rgb = rgb 
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation = cv2.INTER_AREA)
        print('Doing grasp point detection')
        if (self.network == 'GR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

        elif (self.network == 'GGCNN'):
            ##### GGCNN #####
            x = torch.from_numpy(depth.reshape(1,1,self.IMG_WIDTH,self.IMG_WIDTH).astype(np.float32))

        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit() 

        with torch.no_grad():
            xc = x.to(self.device)
            if (self.network == 'GR_ConvNet'):
                ##### GR-ConvNet #####
                pred = self.net.predict(xc)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(pred['pos'],
                                                                pred['cos'],
                                                                pred['sin'],
                                                                pred['width'],
                                                                pixels_max_grasp)
            elif (self.network == 'GGCNN'):
                ##### GGCNN #####
                pred = self.net(xc)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(pred[0],
                                                                pred[1],
                                                                pred[2],
                                                                pred[3],
                                                                pixels_max_grasp)
            else: 
                print ("you need to add your function here!")        
        

        """MASK R CNN IS DOING STUFF HERE"""
        #possible_labels = ["cracker_box", "sugar_box", "tomato_soup_can",
        #                   "mustard_bottle", "gelatin_box", "potted_meat_can"]
        #desired_object = "mustard_bottle"
        #method = 'boundingBox'

        # give maskRCNN the object we want, the rgb image, the size of that image, the q_img from 
        # the grasping network, as well as tell it if  we want to save the output of mask-r-cnn as an image.
        objectFound, q_img = self.maskRCNN(desired_object, maskingMethod, maskRCNN_rgb, img_size, q_img, show_output)


        save_name = None
        if show_output:
            #fig = plt.figure(figsize=(10, 10))
            im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            plot = plot_results(self.fig,
                                rgb_img=im_bgr,
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                depth_img=depth,
                                no_grasps=3,
                                grasp_width_img=width_img)

            if not os.path.exists('network_output'):
                os.mkdir('network_output')
            time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_name = 'network_output/{}'.format(time)
            plot.savefig(save_name + '.png')
            plot.clf()

        print(" ")
        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
        return grasps, save_name, objectFound    # the last one is a variable that is true if the object was detected
        #return grasps, save_name

    def predict_grasp(self, rgb, img_size, depth, n_grasps=1, show_output=False, desired_object='mustard_bottle', maskingMethod='boundingBox'):
        predictions, save_name, objectFound = self.predict(rgb, img_size, depth, n_grasps, show_output, desired_object, maskingMethod)
        grasps = []
        for grasp in predictions:
            x, y, z, roll, opening_len, obj_height = self.grasp_to_robot_frame(grasp, depth)
            grasps.append((x, y, z, roll, opening_len, obj_height))

        return grasps, save_name, objectFound
