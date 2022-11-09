simulation.py has changes in it that allow us to specify the method Mask R CNN should use to mask the q_img of the grasping networks. This can be either:
- 'boundingBox', which just uses the entire rectangle of the bounding box of the object or 
- 'objectMask', which uses just the outline of the object (THIS IS NOT IMPLEMENTED YET)
We can also specify the object we want in simulation,py whenever we call generator.predict_grasp(). The objects we can ask mask-r-cnn for are (only use one!):
- ["cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "gelatin_box", "potted_meat_can"]

