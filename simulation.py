from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, PackPileData, IsolatedObjData, summarize
import numpy as np
import pybullet as p
import argparse
import os
import sys
import cv2
import math
import matplotlib.pyplot as plt
import time
from trained_models.Mask_RCNN.MRCNN import mask_rcnn_size
import json



class GrasppingScenarios():

    def __init__(self, network_model="GGCNN"):
        
        self.network_model = network_model

        if (network_model == "GR_ConvNet"):
            ##### GR-ConvNet #####
            self.IMG_SIZE = 224
            self.network_path = 'trained_models/GR_ConvNet/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
            sys.path.append('trained_models/GR_ConvNet')
        elif (network_model == "GGCNN"):
            ##### GGCNN #####
            self.IMG_SIZE = 300
            self.network_path = 'trained_models/ggcnn/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
            sys.path.append('trained_models/ggcnn')
        else:
            print("The selected network has not been implemented yet!")
            exit() 
        
        
        self.CAM_Z = 1.9
        self.depth_radius = 1
        self.ATTEMPTS = 3
        self.fig = plt.figure(figsize=(10, 10))
       
                
    def draw_predicted_grasp(self,grasps,color = [0,0,1],lineIDs = []):
        x, y, z, yaw, opening_len, obj_height = grasps

        gripper_size = opening_len + 0.02 
        finger_size = 0.075
        # lineIDs = []
        lineIDs.append(p.addUserDebugLine([x, y, z], [x, y, z+0.15],color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    color, lineWidth=6))

        lineIDs.append(p.addUserDebugLine([x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z], 
                                    [x - gripper_size*math.sin(yaw), y - gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        lineIDs.append(p.addUserDebugLine([x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z], 
                                    [x + gripper_size*math.sin(yaw), y + gripper_size*math.cos(yaw), z-finger_size], 
                                    color, lineWidth=6))
        
        return lineIDs
    
    def remove_drawing(self,lineIDs):
        for line in lineIDs:
            p.removeUserDebugItem(line)
    
    def dummy_simulation_steps(self,n):
        for _ in range(n):
            p.stepSimulation()

    def is_there_any_object(self,camera):
        self.dummy_simulation_steps(10)
        rgb, depth, _ = camera.get_cam_img()
        #print ("min RGB = ", rgb.min(), "max RGB = ", rgb.max(), "rgb.avg() = ", np.average(rgb))
        #print ("min depth = ", depth.min(), "max depth = ", depth.max())
        if (depth.max()- depth.min() < 0.0025):
            return False
        else:
            return True
                    
                    
    def isolated_obj_scenario(self, runs, attempts_grasp, method, confidence_threshold, device, vis, output, debug):

        objects = YcbObjects('objects/ycb_objects')
        
        ## reporting the results at the end of experiments in the results folder
        data = IsolatedObjData(objects.obj_names, runs, 'results')

        ## camera settings: cam_pos, cam_target, near, far, size, fov
        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z

        ## camera should be size=500 for mask-rcnn
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (mask_rcnn_size, mask_rcnn_size), 40)
        env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)
        
        objects.shuffle_objects()

        for i in range(runs):
            print("----------- run ", i + 1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")
            print("number of grasp attents: ", attempts_grasp)
            print("method: ", method)
            print("confidence threshold: {:.2f}".format(confidence_threshold))

            for obj_name in objects.obj_names:
                print(" ")
                print("# Scenario object: ", obj_name)

                env.reset_robot()          
                env.remove_all_obj()                        
               
                path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)

                env.load_isolated_obj(path, mod_orn, mod_stiffness)
                self.dummy_simulation_steps(20)

                number_of_failure_grasp = 0
                idx = 0 ## select the best grasp configuration

                while self.is_there_any_object(camera) and (number_of_failure_grasp < attempts_grasp):     
                    
                    bgr, depth, _ = camera.get_cam_img()

                    ##convert BGR to RGB
                    ## rgb image has 500 size for mask-rcnn
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    ## resize depth image
                    depth = cv2.resize(depth, (self.IMG_SIZE, self.IMG_SIZE), interpolation = cv2.INTER_AREA)

                    data.add_try_recog(obj_name)

                    grasps, save_name, objectFound, objectScore = generator.predict_grasp(rgb, self.IMG_SIZE, depth, attempts_grasp, output, 
                        obj_name, method, confidence_threshold)

                    if objectFound: 
                        data.add_succes_recog(obj_name)
                    else:
                        # if object is not found, don't even try
                        self.dummy_simulation_steps(10)
                        print(' Skipping object... \r\n')
                        break

                    # first tries with the best config, if it fails tries the next one
                    if len(grasps) == 0:
                        number_of_failure_grasp += 1 # grasp not found
                        print('Grasp not found. Number of errors: ', number_of_failure_grasp)
                        continue
                    else:
                        if (idx > len(grasps) - 1):  
                            idx = len(grasps) - 1   

                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_grasp(g,color=[1,0,1],lineIDs=LID)
                        time.sleep(0.5)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10)

                    lineIDs = self.draw_predicted_grasp(grasps[idx])

                    x, y, z, yaw, opening_len, obj_height = grasps[idx]
                    succes_grasp, succes_target, object_grasp_id = env.grasp((x, y, z), yaw, opening_len, obj_height)

                    data.add_try(obj_name)
                   
                    if succes_grasp: 
                        data.add_succes_grasp(obj_name)
                    else:
                        number_of_failure_grasp += 1 # grasp not found
                        print('Grasp not found. Number of errors: ', number_of_failure_grasp)
                        continue

                    if succes_target: 
                        data.add_succes_target(obj_name)

                    ## remove visualized grasp configuration 
                    if vis: self.remove_drawing(lineIDs)

                    env.reset_robot()
                    
                    if succes_target:
                        # resets the counter
                        if vis:
                            debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)
                        
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png') 

                        break

                    else:
                        idx +=1                            
                
                        if vis:
                            debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)

        # data.write_json(self.network_model)
        # summarize(data.save_dir, "isolated", runs, attempts, self.network_model, n_objects, method, confidence_threshold)
        return data


    def packed_or_pile_scenario(self, runs, scenario, attempts_grasp, attempts_clear, n_objects, method, confidence_threshold, device, vis, output, debug):

        ## reporting the results at the end of experiments in the results folder
        objects = YcbObjects('objects/ycb_objects')    

        data = PackPileData(objects.obj_names, n_objects, runs, 'results', scenario)

        center_x, center_y, center_z = 0.05, -0.52, self.CAM_Z
        
        camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 0.2, 2.0, (mask_rcnn_size, mask_rcnn_size), 40)
        env = Environment(camera, vis=vis, debug=debug, finger_length=0.06)
        
        generator = GraspGenerator(self.network_path, camera, self.depth_radius, self.fig, self.IMG_SIZE, self.network_model, device)

        for i in range(runs): # number of times everything is repeated        
            
            print("----------- run ", i + 1, " -----------")
            print ("network model = ", self.network_model)
            print ("size of input image (W, H) = (", self.IMG_SIZE," ," ,self.IMG_SIZE, ")")
            print("number of clears allowed: ", attempts_clear)
            print("number of grasp attents: ", attempts_grasp)
            print("number of objects per scenario: ", n_objects)
            print("method: ", method)
            print("confidence threshold: {:.2f}".format(confidence_threshold))
            
            if vis:
                debugID = p.addUserDebugText(f'Experiment {i+1}', [-0.0, -0.9, 0.8], [0,0,255], textSize=2)
                time.sleep(0.5)
                p.removeUserDebugItem(debugID)

            objects.shuffle_objects()
            info = objects.get_n_first_obj_info(n_objects)

            # creates one scenario per object to evaluate its accuracy
            for obj_name in objects.obj_names:
                print(" ")
                print("# Scenario object: ", obj_name)

                env.remove_all_obj()
                env.reset_robot() 
                
                if scenario == "pile":
                    objects_ids = env.create_pile(n_objects, info)
                elif scenario == "packed":
                    objects_ids = env.create_packed(n_objects, info)
                else:
                    print('Quantum bit error')
                    exit()

                objects_dict = dict(zip(objects_ids, objects.obj_names))

                number_of_grasp_failures = 0
                number_of_target_failures = 0
                number_of_objects_cleared = 0
                idx = 0
            
                while self.is_there_any_object(camera) and (number_of_grasp_failures <= attempts_grasp) and (number_of_objects_cleared <= attempts_clear):   
                    print(" ")
                    
                    # 1. Recognize the object with Mask-RCNN
                    data.add_try_recog(obj_name)

                    rgb, depth, _ = camera.get_cam_img()
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                    ## resize depth image
                    depth = cv2.resize(depth, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)

                    grasps, save_name, objectFound, objectScore = generator.predict_grasp(rgb, self.IMG_SIZE, depth, attempts_grasp, output, 
                                                                            obj_name, method, confidence_threshold)

                    if objectFound: 
                        discrimination = True
                        if objectFound > 1:
                            for ii in range(1, objectFound - 1):
                                if (objectScore[0] - objectScore[ii]) <= 0.01: 
                                    discrimination = False # no distinction between desired detected objects
                        
                        if discrimination:
                            print('Object found!')
                            data.add_succes_recog(obj_name)
                        else:
                            print('Several objects with the same score of the {} class were found, skipping...'.format(obj_name))
                            break

                    # 2. Remove either the chosen object or the best one
                    if objectFound: data.add_try(obj_name) # it will try to grap the desired object

                    if len(grasps) == 0:
                        number_of_grasp_failures += 1 # grasp not found

                        print('Grasp not found. Error: {}'.format(number_of_grasp_failures))
                        continue # no objects can be grasped
                    else:
                        if (idx > len(grasps) - 1):  
                            idx = len(grasps) - 1                     

                    if vis:
                        LID =[]
                        for g in grasps:
                            LID = self.draw_predicted_grasp(g, color=[1,0,1],lineIDs=LID)
                        time.sleep(0.5)
                        self.remove_drawing(LID)
                        self.dummy_simulation_steps(10)
                        
                    lineIDs = self.draw_predicted_grasp(grasps[idx])
                    
                    ## perform object grasping and manipulation : 
                    #### succes_grasp means if the grasp was successful, 
                    #### succes_target means if the target object placed in the target basket successfully
                    x, y, z, yaw, opening_len, obj_height = grasps[idx]
                    succes_grasp, succes_target, grasped_obj_id = env.grasp((x, y, z), yaw, opening_len, obj_height)
                    
                    if succes_grasp: 
                        if objectFound: 
                            print('Object was grasped')
                            data.add_succes_grasp(obj_name) 
                        # env.remove_obj(grasped_obj_id)     
                    else:
                        number_of_grasp_failures += 1
                        print('Error grasping the object. Number of errors: {}'.format(number_of_grasp_failures))

                    ## remove visualized grasp configuration 
                    if vis: 
                        self.remove_drawing(lineIDs)

                    env.reset_robot()
                        
                    if succes_target:
                        if objectFound: data.add_succes_target(obj_name)
                        idx = 0 # restarts the optimal selection

                        if vis:
                            debugID = p.addUserDebugText("success", [-0.0, -0.9, 0.8], [0,0.50,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)
                            
                        if save_name is not None:
                            os.rename(save_name + '.png', save_name + f'_SUCCESS_grasp{i}.png')
                            
                    else:
                        number_of_target_failures += 1
                        idx +=1  # it will try with the next config   
                                                
                        if vis:
                            debugID = p.addUserDebugText("failed", [-0.0, -0.9, 0.8], [0.5,0,0], textSize=2)
                            time.sleep(0.25)
                            p.removeUserDebugItem(debugID)

                    if not grasped_obj_id == None:
                        removed_object = objects_dict[grasped_obj_id]
                        print('Object: {} was removed'.format(removed_object))
                        if removed_object == obj_name:
                            if objectFound: data.add_succes_scenario(obj_name)

                            break # the object being tested is no longer on the table
                        else:
                            number_of_objects_cleared += 1
                            print('Number of objects cleared: ', number_of_objects_cleared)
                            
        return data
        
def parse_args():
    parser = argparse.ArgumentParser(description='Grasping demo')

    parser.add_argument('--scenario', type=str, default='isolated', help='Grasping scenario (isolated/packed/pile)')
    parser.add_argument('--network', type=str, default='GR_ConvNet', help='Network model (GR_ConvNet/...)')

    parser.add_argument('--runs', type=int, default=1, help='Number of runs the scenario is executed')
    parser.add_argument('--attempts_clear', type=int, default=3, help='Number of objects that may be cleared before detecting the desired one')
    parser.add_argument('--attempts_grasp', type=int, default=3, help='Number of attempts in case grasping failed')

    parser.add_argument('--save-network-output', dest='output', type=bool, default=False,
                        help='Save network output (True/False)')

    parser.add_argument('--device', type=str, default='cpu', help='device (cpu/gpu)')
    parser.add_argument('--vis', type=bool, default=True, help='vis (True/False)')
    parser.add_argument('--report', type=bool, default=True, help='report (True/False)')

    parser.add_argument('--n_objects', type=int, default=-1, help='number of objects per scenario')
    parser.add_argument('--confidence_threshold', type=float, default=0.8, help='Labelled objects with a lower confidence value than this are disarded')
    parser.add_argument('--method', type=str, default='boundingBox', help='Method use to mask the q_img in MASK-RCNN (boundingBox, objectMask)')

    parser.add_argument('--flag_test', type=int, default=0, help='If true, then the parameters introduced are ignored and the model is tested')
    parser.add_argument('--out_path', type=str, default='results/', help='Output path for the summaries')
                        
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()

    output = args.output
    runs = args.runs
    attempts_clear = args.attempts_clear
    attempts_grasp = args.attempts_grasp
    device = args.device
    vis = args.vis
    report = args.report
    n_objects = args.n_objects
    method = args.method
    confidence_threshold = args.confidence_threshold
    flag_test = args.flag_test

    out_path = args.out_path

    grasp = GrasppingScenarios(args.network)

    const_runs = 1

    if flag_test:
        if False:
            scenarios = []; runs = []; methods = []; n_objects = []; thresholds = []; attempts_grasps = []; out_results = []
            
            # conf_methods = ['boundingBox', 'objectMask']; conf_objects = [1]; conf_threshold = [0.7, 0.8, 0.9]; attempts = [1, 2, 3]
            conf_methods = ['boundingBox']; conf_objects = [1]; conf_threshold = [0.7]; conf_attempts = [1]
            for method in conf_methods:
                for n_obj in conf_objects:
                    for thres in conf_threshold:
                        for n_atemp in conf_attempts:

                            scenarios.append('isolated')
                            runs.append(const_runs)
                            methods.append(method)
                            n_objects.append(n_obj)
                            thresholds.append(thres)
                            attempts_grasps.append(n_atemp)

                            results = grasp.isolated_obj_scenario(const_runs, n_atemp, method, thres, device, vis, output=output, debug=False)

                            result_dict = results.to_dict()
                            out_results.append(result_dict)

            out_dict = {
                'scenarios': scenarios,
                'runs': runs,
                'methods': methods,
                'n_objects': n_objects,
                'thresholds': thresholds,
                'attempts_grasps': attempts_grasps,
                'out_results': out_results
            }

            out_file = out_path + "results_isolated.json"
            with open(out_file, "w") as outfile:
                json.dump(out_dict, outfile)
                outfile.close()
                print(" Results stored in : ", out_file)

        if True:
            scenarios = []; runs = []; methods = []; n_objects = []; thresholds = []; attempts_grasps = []; attempts_clears = []; out_results = []
            
            # conf_scenarios = ['packed', 'pile']; conf_methods = ['boundingBox', 'objectMask']; conf_objects = [2, 3, 4, 55]; conf_threshold = [0.7, 0.8, 0.9]; conf_attempt_grasps = [1, 2, 3]; conf_attempt_clears = [0, 1, 2]
            conf_scenarios = ['pile']; conf_methods = ['boundingBox']; conf_objects = [5]; conf_threshold = [0.7]; conf_attempt_grasps = [0]; conf_attempt_clears = [0]
            for scenario in conf_scenarios:
                for method in conf_methods:
                    for n_obj in conf_objects:
                        for thres in conf_threshold:
                            for attemp_grasp in conf_attempt_grasps:
                                for attemp_clear in conf_attempt_clears:

                                    scenarios.append(scenario)
                                    runs.append(const_runs)
                                    methods.append(method)
                                    n_objects.append(n_obj)
                                    thresholds.append(thres)
                                    attempts_grasps.append(attemp_grasp)
                                    attempts_clears.append(attemp_clear)

                                    results = grasp.packed_or_pile_scenario(const_runs, scenario, attemp_grasp, attemp_clear, n_obj, method, thres, 
                                                        device, vis, output=output, debug=False)

                                    result_dict = results.to_dict()
                                    out_results.append(result_dict)

            out_dict = {
                'scenarios': scenarios,
                'runs': runs,
                'methods': methods,
                'n_objects': n_objects,
                'thresholds': thresholds,
                'attempts_grasps': attempts_grasps,
                'attempts_clears': attempts_clears,
                'out_results': out_results
            }

            out_file = out_path + "results_packed_pile.json"
            with open(out_file, "w") as outfile:
                json.dump(out_dict, outfile)
                outfile.close()
                print(" Results stored in : ", out_file)

    else:
        if args.scenario == 'isolated':
            results = grasp.isolated_obj_scenario(runs, attempts_grasp, method, confidence_threshold, device, vis, output=output, debug=False)

        elif args.scenario == 'packed' or args.scenario == 'pile':
            if n_objects <= 0: n_objects = 5
            results = grasp.packed_or_pile_scenario(runs, args.scenario, attempts_grasp, attempts_clear, n_objects, method, confidence_threshold, 
                                                    device, vis, output=output, debug=False)
        else:
            print('Scenario: {} not valid'.format(args.scenario))
            exit()
