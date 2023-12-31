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
* This section notes the steps to open a video for bounding box creation and processing. Also includes other features of the application. 

1. Run the interface
   ```sh
   Run python .\bounding_box_interface_to_use.py 
   ```
2. File explorer should open, and select the wanted video file to add bounding boxes.  
4. Select 'Open Video file' in the top menu to open the first frame of the video
5. Add wanted bounding boxes, note rules for bounding boxes
   * Add straight by selecting the 'rectangle' option and box direction, and create the bounding box by dragging across the screen.
   * Add a curve by selecting the 'curve' option and box direction, and create the bounding box by dragging across the screen. Select the start and drag to the end of the curve.
6. Select 'Process Bounding Boxes' to display the video with bounding boxes in a separate window.
* Note there can be no interaction with the bounding box interface as long as this window is broken
* To EXIT the bounding box window, press ESC with the window selected. 

Other features
* Undo button: Undo the last bounding box
* Clear button: Clears all of the bounding boxes
* Save bounding boxes: Save the current bounding boxes to an empty JSON file (create an empty file with .json extension first)
* Load bounding boxes: Opens the file explorer, and select a VALID JSON file to open and load the bounding boxes. 


# Algorithm DropShop.py
## Summary:
Every frame in the video will now be analyzed with the machine learning (ML) model and given an array of detections. 
Each detection is mapped to the existing Droplets and updated to those droplets' positions. 
Green boxes are straight segments, Blue boxes are Curve segments, red dots are points along the curve to calculate the quadratic coefficients a, b, c,
black boxes are dispensers, purple boxes are ML model's detections (the top left-hand number is ID, the right-hand number is confidence),
Cyan/Aqua boxes inside the purple boxes are Python Python-initialized droplets to store data. 
If a detection is missed then predict where it'll be using the fact the Droplet is in a straight or curve traveling in 1 direction
### Important Note get_droplet_on_screen() is hard coded to initialize droplets at time T and location (x, y) referring to a dispenser. So varies by video
1. Pass in weights_path which is a file of weights trained with YoloV8 and a video path to main()
    ```if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    main("runs/detect/train10/weights/best.pt", "droplet_videos/video_data_Rainbow 11-11-22.m4v")
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    ```
2. The main function initializes variables and runs the core logic
all_droplets stores every droplet in the course at any given point.
course is a Path object that stores the segments in order. The Path object has a function to add new segments and add droplets to each nested segment's queue.

    ```class Path():
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
              print([drop.id for drop in self.segments_in_order[droplet.current_section].queue])
    ```
    
3. Each Segment is either a Straight or a Curve and each one holds a data structure that helps store using the top left corner point and bottom right-hand corner point. 
Straights are simple only having an add droplet and remove droplet feature with most parameters passed in by the User. 

    class Straight():
          def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
              self.top_left = point1
              self.bottom_right = point2
              self.direction = direction
              self.queue = set()
             
          def add_droplet(self, droplet: Droplet) -> None:
              '''Add a droplet to the queue'''
              self.queue.add(droplet)
          
          def remove_droplet(self, droplet: Droplet) -> None:
              '''Removes a droplet from this segments queue'''
              self.queue.remove(droplet)
              
4. Curves are more complex it needs a corresponding start, middle, and endpoint which calls a quadratic function to solve for a, b, and c in ax^2 + bx + c and a function
predict y that helps infer the location of the droplet

    class Curve():
          def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
              '''Initialize a curve's box and it's direction. Assuming a start, middle, end point are provided.
              Initialize a tuple that holds the coefficients to a quadratic formula (a, b, c) for the respective
              f(x) = ax^2 + bx + c
              '''
              self.top_left = point1
              self.bottom_right = point2
              self.direction = direction 
              self.start = None
              self.mid = None
              self.end = None
              self.queue = set()
              self.quadratic_coef = None #Holds a, b, c coefficients of quadratic formula
      
          def add_droplet(self, droplet: Droplet) -> None:
              '''Add a droplet to the queue'''
              self.queue.add(droplet)
          
          def remove_droplet(self, droplet: Droplet) -> None:
              '''Remove a droplet to the queue'''
              self.queue.remove(droplet)
          
          def add_sme(self, s: (int, int), m: (int, int), e: (int, int)) -> None:
              '''Adds the start middle end points and gets then uses those points to get the coefficients'''
              self.start = s
              self.mid = m
              self.end = e
              self.quadratic_coef = self.get_quadratic(s, m, e)
          
          def get_quadratic(self, s: (int, int), m: (int, int), e: (int, int)) -> (int, int, int):
              '''Returns a tuple that holds the coefficients to a quadratic formula (a, b, c) for the respective
              f(x) = ax^2 + bx + c '''
              x_1 = s[0]
              x_2 = m[0]
              x_3 = e[0]
              y_1 = s[1]
              y_2 = m[1]
              y_3 = e[1]
      
              a = y_1/((x_1-x_2)*(x_1-x_3)) + y_2/((x_2-x_1)*(x_2-x_3)) + y_3/((x_3-x_1)*(x_3-x_2))
      
              b = (-y_1*(x_2+x_3)/((x_1-x_2)*(x_1-x_3))
                  -y_2*(x_1+x_3)/((x_2-x_1)*(x_2-x_3))
                  -y_3*(x_1+x_2)/((x_3-x_1)*(x_3-x_2)))
      
              c = (y_1*x_2*x_3/((x_1-x_2)*(x_1-x_3))
                  +y_2*x_1*x_3/((x_2-x_1)*(x_2-x_3))
                  +y_3*x_1*x_2/((x_3-x_1)*(x_3-x_2)))
              return a,b,c
      
          def predict_y(self, x: int) -> int:
              '''Given an integer x return the respective y value from the quadratic formula'''
              a, b, c = self.quadratic_coef
              return a * (x ** 2) + b * x + c

5. Every Python object droplet holds its id, (x, y), trajectory, what section it's currently in, the last time it was seen, and a different speed for curves.
Its primary function is
### update_position which updates its position this is the logic in the case it's not detected it updates depending on where it was in the curve.
### update_section When the droplet is detected outside of the section it's currently in move the data over to the next segment and the next segment.
For example, if a droplet was in S0 and saw in S1 add it to S1 and S2. The design choice for this was to allow for a seamless transition in data since there was too much
variability in accuracy this provided a generally confident way that the Droplet would at least be seen in S1 or S2 and update. This makes the algorithm heavily reliant on the model
being accurate to a certain degree.
### update_last_seen which dynamically updates the speed of the droplet using last seen to calculate it

    class Droplet():
        def __init__(self, id, x: int = None, y:int = None, trajectory: int = 1, current_section: int = 0) -> None:
            '''Initialize Droplet Object'''
            self.id = id
            self.x = x
            self.y = y
            self.trajectory = trajectory
            self.current_section = current_section
            self.last_detection = None
            self.curve_speed = trajectory
    
        def update_position(self, course: Path, droplet) -> (int, int):
            segment = course.segments_in_order[self.current_section]
            direction_x, direction_y = segment.direction
            if isinstance(segment, Straight):
    
                # self.x += (self.trajectory * direction_x)
                # #slope = (segment.top_left[1] - segment.top_right[1])/(segment.top_left[0] - segment.top_right[0]) #ideally most  cases slope is 0
                # slope = 0 
                # self.y += slope
                if direction_x and not direction_y:
                    self.x += (self.trajectory * direction_x)
                else:
                    self.y += (self.trajectory * direction_y)
            else:
                try:
                    try:
                        self.x += (self.curve_speed * direction_x)
                    except AttributeError:
                        print("Occured o nself.x")
                    self.y = segment.predict_y(self.x)
                except AttributeError:
                    print("Occurred here")
            self.update_section(course, droplet)
            return (self.x, self.y)
        
        def update_section(self, course: Path, droplet) -> None:
            segment = course.segments_in_order[self.current_section]
            left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
            if self.x < left or self.x > right or self.y < top or self.y > bot:
                if self.current_section < len(course.segments_in_order):
    
                    #Error Probably Occurring Here
                    course.segments_in_order[self.current_section].remove_droplet(droplet)
                    self.current_section += 1
                    course.segments_in_order[self.current_section].add_droplet(droplet)
                    if self.current_section + 1 < len(course.segments_in_order):
                        course.segments_in_order[self.current_section + 1].add_droplet(droplet)
        
        def update_last_seen(self, mid : (int, int), t : int, x_y_map: {(int, int): Path}, speed_threshold : int) -> None:
            self.x = mid[0]
            self.y = mid[1]
    
            if not self.last_detection:
                self.last_detection = (mid, t)
                return
            else:
                if isinstance(x_y_map[mid], Straight):
                    last_x, curr_x, last_t = self.last_detection[0][0], mid[0], self.last_detection[1]
                    if t != last_t: #This line prevents Zero Division Error
                        new_trajectory =  max((last_x - curr_x), (curr_x - last_x))//max((last_t - t), (t - last_t))
                        if new_trajectory and new_trajectory <= speed_threshold:
                            self.trajectory = new_trajectory
                else:
                    current_curve = x_y_map[mid]
                    middle_curve_x = current_curve.mid[0]
                    start_x, end_x = current_curve.start[0], current_curve.end[0]
                    total_length = abs((start_x - end_x))
                    proximity_to_center = abs(middle_curve_x - self.x)
                    if proximity_to_center/total_length * self.curve_speed >= 0.3: 
                        self.curve_speed *= proximity_to_center/total_length 
                self.last_detection = (mid, t)



<p align="right">(<a href="#readme-top">back to top</a>)</p>




  
