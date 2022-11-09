import argparse
import random
import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from datetime import datetime
from torchvision.transforms import transforms as transforms


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# default objects list to use by the model
ycb = [
    "cracker_box", "sugar_box", "tomato_soup_can",
    "mustard_bottle", "gelatin_box", "potted_meat_can"
]

ycb_original_labels = ["CrackerBox", "xx", "TomatoSoupCan", 
    "MustardBottle", "GelatinBox", "PottedMeatCan"
]

COLORS = np.random.uniform(0, 255, size=(len(ycb), 3))


class MRCNN():
    # Base class for using the MASK-RCNN model
    def __init__(self):
        # load pretrained model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        num_classes = len(ycb) + 1
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # creates internal mapping dictionary
        self.dict = dict(zip(ycb_original_labels, ycb))

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
        weights_path = "trained_models/Mask_RCNN/clutter_maskrcnn_model.pt"
        self.device = torch.device('cpu')
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))    # load weights
        self.model.to(self.device).eval() 

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def convert_labels_to_internal_rep(self, labels):
        labels = [self.dict[label] for label in labels]
        return labels

    def convert_labels_to_external_rep(self, labels):
        labels = [list(self.dict.keys())[list(self.dict.values()).index(label)] for label in labels]
        return labels

    def get_outputs(self, 
        image, # RGB-image of the scene
        threshold, # Confidence value in [0, 1]
        ):

        with torch.no_grad():
            # forward pass of the image through the modle
            outputs = self.model(image)
        
        # get all the scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        
        # get the masks
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()

        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
        
        # get the classes labels
        labels = [ycb[i] for i in outputs[0]['labels']]

        # discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]

        # discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]

        # discard labels below threshold value
        labels = labels[:thresholded_preds_count]

        scores = scores[:thresholded_preds_count]

        return masks, boxes, labels, scores


        
    def draw_segmentation_map(self, image, masks, boxes, labels):
        # alpha = 1 
        # beta = 0.6 # transparency for the segmentation map
        # gamma = 0 # scalar added to each sum
        for i in range(len(masks)):
            red_map = np.zeros_like((masks[i])).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # apply a random colour mask to each object
            color = COLORS[random.randrange(0, len(COLORS))]
            # color_false = (np.ones(shape=(1, 3)) * 2.5)[0] # avoids binary images

            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
            # red_map[masks[i] == 0], green_map[masks[i] == 0], blue_map[masks[i] == 0]  = color_false
            
            # combine all the masks into a single image
            # print("Red:", red_map)
            # print("Green:", green_map)
            # print("Blue:", blue_map)
            # segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            #convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGN to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            # cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)
            # put the label text above the objects
            cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                        thickness=2, lineType=cv2.LINE_AA)
        
        return image


    def segmentImage(self, image, desired_label, confidence_threshold=0.85, save_output_image=False):
        orig_image = image.copy()
        # transform the image
        image = self.transform(image)
        # add a batch dimension
        image = image.unsqueeze(0).to(self.device)

        masks, boxes, labels, scores = self.get_outputs(image, confidence_threshold)

        labels = self.convert_labels_to_external_rep(labels)
        
        # gather all the boxes of the desired objects
        outs = ["{}->{:.2f}".format(ll, sc) for ll, sc in zip(labels, scores)]
        print("Segmented objects: ", ", ".join(outs))

        box = []; mask = []
        for label, ii in zip(labels, range(len(labels))):
            if label == desired_label:
                box.append(boxes[ii]); mask.append(masks[ii])

        print('Number of desired_objects found: {}'.format(len(box)))

                
        if len(masks) > 0:
            result_img = self.draw_segmentation_map(orig_image, masks, boxes, labels)
            # save the segmented image output
            if save_output_image:
                if not os.path.exists('network_output'):
                    os.mkdir('network_output')
                time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                filename = "network_output/" + str(time) + "_maskrcnnOutput.png"
                cv2.imwrite(filename, result_img)
            #cv2.imshow('Segmented image', result) # not so great to run

        return box, mask



    def predict(self, desired_object, method, rgb, q_img, img_ratio=1.0, conf_threshold=0.8, show_output=True):
        print('Predict - MASK-RCNN')     
        objectBox, objectMask = self.segmentImage(rgb, desired_object, conf_threshold, save_output_image=show_output) 

        object_found = False

        ## resize the object box
        if len(objectBox) > 0:
            object_found = True

            if method == 'boundingBox':
                # unpack box edges
                print("MASKING Q-IMAGE")
                left_bound, top_bound = objectBox[0][0] # TODO FILTER THE FOUND OBJECTS IN CASE THERE ARE MORE THAN 1
                right_bound, bottom_bound = objectBox[0][1]

                left_bound = round(left_bound / img_ratio)
                top_bound = round(top_bound / img_ratio)
                right_bound = round(right_bound / img_ratio)
                bottom_bound = round(bottom_bound / img_ratio)

                q_img[  :top_bound,      :] = 0         # everything above the box
                q_img[bottom_bound:,      :] = 0         # everything below the box
                q_img[:,             :left_bound] = 0    # left of the box
                q_img[:,             right_bound:] = 0   # right of the box

            elif method == 'objectMask':
                print("object mask outline not implemented yet")
        else:
            print("object not found!")
        
        return object_found, q_img
