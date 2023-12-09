<a name="readme-top"></a>
# Droptop Interface with Yolo V8 CVS 



<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Description">About The Project</a>
    </li>
    <li>
      <a href="#Features">Features</a>
    </li>
    <li><a href="#Getting Started">Getting Started</a></li>
  </ol>
</details>

# Description 
  Proposed improvement of the original Droptop computer vision system. Adds functionality of the current computer vision system with droplet labeling for droplet reacquisition after loss. Includes basic interface for bounding box manipulation, saving, and loading.   


# Features
  ## Interface 
  * GUI provides users the ability to load videos and create bounding boxes for the computer vision system. Includes: video loading, bounding box saving/loading, bounding box processing, setting attributes bounding box type (curve or straight) and direction, clearing of bounding boxes, and undo. Provides a section to display all bounding boxes. 
  
  ## Computer vision system 
  * CVS uses YOLO v8 for droplet detection and uses models trained with Roboflow (best.pt) 

  <p align="right">(<a href="#readme-top">back to top</a>)</p>

  
# Getting Started 
Instructions to get the code to start working :) 

## Prerequisites
* using npm to install the requirements.txt for the project. Need a basic concept of git and python to use. 
```sh
  npm install npm@latest -g
```
## Installation
1. Clone the repo
   ```sh
   git clone https://github.com/ElysonEva/117-droptop-project.git
   ```
2. Install NPM packages (requirements.txt) 
   ```sh
   npm install
   ```
   <p align="right">(<a href="#readme-top">back to top</a>)</p>
## Usage 
* This section notes the steps on how to open a video for bounding box creation and processing.

1. Run the interface
   ```sh
   Run python .\bounding_box_interface_to_use.py 
   ```
2. Select Open Video file with File Explorer
3. Add wanted bounding boxes, note rules for bounding boxes
   * Add straight by creating the bounding box
   * Add a curve by creating the bounding box for the corner and selecting the start and end of the curve.
   * Select the direction of the bounding box.  
4. Select 'Process Bounding Boxes'

# Algorithm DropShop.py
## Summary:
Every frame in the video will now be analyzed with the machine learning (ML) model and given an array of detections. 
Each detection is mapped to the existing Droplets and updated to those droplets' positions. 
Green boxes are straight segments, Blue boxes are Curve segments, red dots are points along the curve to calculate the quadratic coefficients a, b, c,
black boxes are dispensers, purple boxes are ML model's detections (the top left-hand number is ID, the right-hand number is confidence),
Cyan/Aqua boxes inside the purple boxes are Python Python-initialized droplets to store data.
1. Pass in weights_path which is a file of weights trained with YoloV8 and a video path to main() 
  ```
if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    main("runs/detect/train10/weights/best.pt", "droplet_videos/video_data_Rainbow 11-11-22.m4v")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")```
2. The main function initializes variables and runs the core logic
all_droplets stores every droplet in the course at any given point.
course is a Path object that stores the segments in order
```
class Path():
    def __init__(self) -> None:
        self.segments_in_order = []
    
    def add_segment(self, new_segment) -> None:
        '''Adds a segment of the course into the Paths array'''
        self.segments_in_order.append(new_segment)

    def add_droplet_to_queues(self, droplet) -> None:
        '''Adds droplets to the corresponding queue in each segment. If the segment isn't last in the list then
        add it to the correct one and the next one. That way we can infer will it will be and remove it accordingly'''
        length = len(self.segments_in_order)
        if length > 1 and droplet.current_section + 1 < length:
            self.segments_in_order[droplet.current_section + 1].add_droplet(droplet)
        self.segments_in_order[droplet.current_section].add_droplet(droplet)
        print([drop.id for drop in self.segments_in_order[droplet.current_section].queue])```
      


<p align="right">(<a href="#readme-top">back to top</a>)</p>




  
