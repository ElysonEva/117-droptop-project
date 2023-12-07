# Droptop Interface with Yolo V8

# Description 
  Proposed improvement of the original Droptop computer vision system. Adds functionality of the current computer vision system with droplet labeling for droplet reacquisition after loss. Includes basic interface for bounding box manipulation, saving, and loading.   

# Features
  ## Interface 
  GUI provides users the ability to load videos and create bounding boxes for the computer vision system. Includes: video loading, bounding box saving/loading, bounding box processing, setting attributes bounding box type (curve or straight) and direction, clearing of bounding boxes, and undo. Provides a section to display all bounding boxes. 
  ## Computer vision system 
  CVS uses YOLO v8 for droplet detection and uses models trained with Roboflow (best.pt) 
  
