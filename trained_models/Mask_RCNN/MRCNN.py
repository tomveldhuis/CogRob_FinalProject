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


ycb = [
    "cracker_box", "sugar_box", "tomato_soup_can",
    "mustard_bottle", "gelatin_box", "potted_meat_can"
]
COLORS = np.random.uniform(0, 255, size=(len(ycb), 3))


# for loading the pretrained model
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



def get_outputs(image, model, threshold):

    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [ycb[i] for i in outputs[0]['labels']]
    return masks, boxes, labels

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a random colour mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        #cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image


def segmentImage(image, desired_label, confidence_threshold=0.85, save_output_image=False):
    # initialise model
    weights_path = "/home/cognitiverobotics/CogRob_FinalProject/trained_models/Mask_RCNN/clutter_maskrcnn_model.pt"
    num_classes = len(ycb)+1
    model = get_instance_segmentation_model(num_classes)
    device = torch.device('cpu')
    model.load_state_dict(
        torch.load(weights_path, map_location=device))    # load weights
    model.to(device).eval() 

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    orig_image = image.copy()
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    masks, boxes, labels = get_outputs(image, model, confidence_threshold)
    
    result = draw_segmentation_map(orig_image, masks, boxes, labels)
    
    # save the segmented image output
    if save_output_image:
        if not os.path.exists('network_output'):
            os.mkdir('network_output')
        time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        filename = "network_output/" + str(time) + "_maskrcnnOutput.png"
        cv2.imwrite(filename, result)
    #cv2.imshow('Segmented image', result) # not so great to run

    # shorten the list of labels to the number of boxes we have
    numLabels = len(boxes)
    labels = labels[:numLabels]
    
    # gather all the boxes of the desired objects
    print(labels)
    # see if the desired object is found
    try:
        boxNumber = labels.index(desired_label)
        box = boxes[boxNumber]
        mask = masks[boxNumber]
    except: # not found
        return False, False

    return box, mask
