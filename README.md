simulation.py has changes in it that allow us to specify the method Mask R CNN should use to mask the q_img of the grasping networks. This can be either:
- 'boundingBox', which just uses the entire rectangle of the bounding box of the object or 
- 'objectMask', which uses just the outline of the object 

We can also specify the object we want in simulation,py whenever we call generator.predict_grasp(). The objects we can ask mask-r-cnn for are:
- ["cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle", "gelatin_box", "potted_meat_can"]

The actual code for GGCNN is not present in this repo, as it is linked from the original dougsm/ggcnn repo and will thus have to be downloaded from there. The weights will also have to be downloaded, but are the same "ggcnn_epoch_23_cornell" weights used in Assigment 2 (which is also on the dougsm/ggcnn repo).


RUNNING SIMULATIONS
To run the simulations, from the terminal run: "./start.sh" while in the project directory.  --flag_test=1 will initialise the simulation in our 'testing' mode. The parameters can be adjusted in line 471 of the simulation.py file (to run different masking methods, trial numbers, etc., an example is seen in the line above, but we each run one setting seperately and combine the data later because it takes very long to run all of the trials). The parameters on line 471 are defined below. When  --flag_test=1, then the parameters inside of start.sh are ignored.

simulation.py can be called either in test mode (--flag_test=1) or single mode (--flag_test=0). 
- In single mode, the next parameters may be introduced (otherwise it will use default values):
    - scenario: isolated, packed, pile
    - runs: number of experiments to run 
    - n_objects: for the packed/pile scenarios, it defines the number of objects to use. Maximum is 5.
    - confidence_threshold: defines the minimum confidence value that Mask-RCNN will use to identify an object
    - attempts_clear: it defines the maximum number of objects that may be removed in the packed/pile scenarios when Mask-RCNN does not detect an object. In case the object removed is the user specifided one, then the scenario stops
    - attempts_grasp: maximum number of attempts to grasp objects. In the packed/pile scenarios it defines the total number of attempts, not only for a single object.

- In test mode by defining flag_test = 1. In this case it is performed a grid_search with the parameters of the model bor the three scenarios. The results are stored in out_path


