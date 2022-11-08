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

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Banana, MustardBottle, TennisBall, Scissors
obj_names = [
    '__background__', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'sports ball',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'bottle', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'banana', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
    'N/A', 'N/A', 'scissors', 'N/A', 'N/A', 'N/A'
]


COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold):
    coco_names = obj_names

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
    # get the classes labelss
    labels = [coco_names[i] for i in outputs[0]['labels']]
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
        #segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
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


def segmentImage(image, label, confidence_threshold=0.85, save_output_image=False):
    # initialize the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                            num_classes=91)
    # set the computation device
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
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

    # see if the desired object is found
    try:  
        boxNumber = labels.index(label)
    except: # not found
        return False

    return boxes[boxNumber] 



def qImgSegmenter(orig_image, box):
    data_address = orig_image.load()
    for y in range(orig_image.size[1]):
        for x in range(orig_image.size[0]):
            if (x < box[0][0] or x > box[1][0]) or (y < box[0][1] or y > box[1][1]):
                data_address[x, y] = (0,0,0)
        return data_address