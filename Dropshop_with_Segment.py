import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv
import os
import math
from random import randint
import time

class Path():
    def __init__(self):
        '''An array that'll hold each straight and curve in the order the droplets will enter them'''
        self.path = []

    def add_to_path(self, new_section):
        '''Add a section of the path to the Path Object'''
        self.path.append(new_section)

    def print_paths(self):
        '''Print the paths' (x,y) and direction'''
        for path in self.path:
            print(path.top_left, path.bottom_right, path.direction)

class Droplet():
    def __init__(self, id, x: int = None, y:int = None, trajectory: int = 1, current_section: int = 0) -> None:
        '''Initialize Droplet Object'''
        self.id = id
        self.x = x
        self.y = y
        self.trajectory = trajectory
        self.current_section = current_section

    def update_position(self, path: Path) -> None:
        '''Update a droplets position using the assumption of what direction it's traveling in from it's corresponding path.'''
        segment = path.path[self.current_section]
        direction_x, direction_y = segment.direction
        self.x += (self.trajectory * direction_x)
        self.y += (self.trajectory * direction_y)
        return (self.x, self.y)
    
    def update_section(self, path: Path) -> None:
        '''Update which section of the path the droplet is in'''
        segment = path.path[self.current_section]
        left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
        if self.x < left or self.x > right or self.y < top or self.y > bot:
            if self.current_section < len(path.path):
                self.current_section += 1

    # def update_trajectory(self):
    # Trajectory needs to be updated periodically using Curve's start, middle, end, Straight, and where it was previously in the frame



class Straight():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        '''Initialize a straight Box and it's direction'''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction
        self.queue = set()

    #These functions are a work in progress as of 11:42pm 11/07/2023

    def add_droplet(self, droplet: Droplet):
        self.queue.add(droplet)
    
    def remove_droplet(self, droplet: Droplet):
        self.queue.remove(droplet)
        
class Curve():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        '''Initialize a curve's box and it's direction'''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction 

        #These variables and following functions are a work in progress as of 11:42pm 11/07/2023
        self.start = None
        self.mid = None
        self.end = None
        self.queue = set()

    def add_droplet(self, droplet: Droplet):
        self.queue.add(droplet)
    
    def remove_droplet(self, droplet: Droplet):
        self.queue.remove(droplet)


def check_file(path):
    '''Takes a path and checks if it's in the directory'''
    if os.path.isfile(path):
        print("Works")
    else:
        print("Failed")

def load_data():
    '''Load the data into your directory if you don't have the dataset yet'''
    rf = Roboflow(api_key="Izum4d9L9gLJ14xMz3sx")
    project = rf.workspace("dropshop-froos").project("dropshop")
    dataset = project.version(1).download("yolov8")

def get_mid_point(xone: int, yone: int, xtwo: int, ytwo: int) -> (int, int):
    '''Take two corners and return the middle of the two points'''
    return ((xone + xtwo)//2, (yone + ytwo)//2)

def give_me_a_small_box(point: (int, int)) -> ((int, int), (int, int)):
    '''Creates a small box for open CV to generate a bounding box a round a point'''
    #Open CV doesn't support float objects
    return (int(point[0] - 2), int(point[1] - 2)),(int(point[0] + 2), int(point[1] + 2))

def label_path(frame) -> None:
    '''Draws bounding boxes on Curves for now this is assumed given'''
    cv2.rectangle(frame, (75, 50), (460, 70), (0, 255, 0), 2) #First Straight
    cv2.rectangle(frame, (25, 50), (75, 110), (0, 200, 0), 2) #First Curve
    cv2.rectangle(frame, (45, 110), (60, 160), (0, 255, 0), 2) #Second Straight from First Curve to Second Curve # First Vertical
    cv2.rectangle(frame, (40, 160), (100, 205), (0, 200, 0), 2) #Second Curve
    cv2.rectangle(frame, (100, 180), (530, 205), (0, 255, 0), 2) #Third Straight

def label_curves(frame) -> None:
    '''Draw the bounding Boxes for the curvers and their Start, Middle, End'''
    start1_left, start1_right = give_me_a_small_box((75, 60))
    cv2.rectangle(frame, start1_left, start1_right, (0, 0, 200), 2) #First Curve
    
    start1_m_l, start1_m_r = give_me_a_small_box((60, 80))
    cv2.rectangle(frame, start1_m_l, start1_m_r, (0, 0, 200), 2)

    start1_e_l, start1_e_r = give_me_a_small_box((52, 110))
    cv2.rectangle(frame, start1_e_l, start1_e_r, (0, 0, 200), 2)

    start2_left, start2_right = give_me_a_small_box((50, 160))
    cv2.rectangle(frame, start2_left, start2_right, (0, 0, 200), 2) #First Curve
    
    start2_m_l, start2_m_r = give_me_a_small_box((70, 190))
    cv2.rectangle(frame, start2_m_l, start2_m_r, (0, 0, 200), 2)

    start2_e_l, start2_e_r = give_me_a_small_box((100, 195))
    cv2.rectangle(frame, start2_e_l, start2_e_r, (0, 0, 200), 2)

    #cv2.rectangle(frame, (25, 50), (75, 110), (0, 200, 0), 2) #First Curve
    # cv2.rectangle(frame, (40, 160), (100, 205), (0, 200, 0), 2) #Second Curve


def where_droplets_should_start(frame) -> None:
    '''Draws a bounding box in front of dispenser location'''
    cv2.rectangle(frame, (445, 55), (455, 65), (255, 0, 0), 2) #Droplet 1, 4, 5, 6
    cv2.rectangle(frame, (315, 55), (325, 65), (255, 0, 0), 2) #Droplet 2, 3

def get_droplets_on_screen(t : int, num_droplets: int, drops:[Droplet]) -> int:
    '''Initializes Droplet objects this is assumed I know it before hand T == Frame they appear in'''
    if t == 1:
        #(450.0, 60.0) 3
        drops.append(Droplet(1, 450, 60, 2))
        return 1
    elif t == 114:
        #(316.0, 62.0) 114
        drops.append(Droplet(2, 315, 60, 1))
        return 2
    elif t == 147:
        #(316.0, 62.0) 114
        drops.append(Droplet(3, 315, 60, 1))
        return 3
    elif t == 152:
        #(449.0, 60.0) 148
        drops.append(Droplet(4, 450, 60, 1))
        return 4
    elif t == 185:
        #(448.0, 60.0) 184
        drops.append(Droplet(5, 450, 60, .5))
        return 5
    elif t == 222:
        #(449.0, 60.0) 148
        drops.append(Droplet(6, 450, 60, .5))
        return 6
    else:
        return num_droplets

def build_path() -> Path:
    '''This builds the Path object assuming I know the path before hand'''
    path = Path()

    straight1 = Straight((75, 50), (460, 70), (-1, 0)) #Left
    path.add_to_path(straight1)

    curve1 = Curve((25, 50), (75, 110), (-1, 1)) #Left and Down
    path.add_to_path(curve1)

    straight2 = Straight((45, 110), (60, 160), (0, 1)) #Down
    path.add_to_path(straight2)

    curve2 = Curve((40, 160), (100, 205), (1, 1)) #Right and Down
    path.add_to_path(curve2)

    straight3 = Straight((100, 180), (530, 205), (1, 0))
    path.add_to_path(straight3)

    return path

def box_drops(drops: [Droplet], frame) -> None:
    '''This boxs the Droplets I know about'''
    for drop in drops:
        left_predict, right_predict = give_me_a_small_box((drop.x, drop.y))
        cv2.rectangle(frame, left_predict, right_predict, (100, 0, 0), 4)
        # print(drop.x, drop.y, drop.id, drop.current_section)

def get_distance(point1: (int, int), point2: (int, int)) -> float:
    '''Distance formula between two points'''
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)** 2 + (y2 - y1) ** 2)

def find_closest_droplet(drops: [Droplet], mid:(int, int), found: set) -> Droplet:
    '''Finds the closest dropets this is a Brute Force Attempt'''
    #Note any Zeros are being graphed are when # of detections > droplets. And I haven't handled that case yet
    closest = float('inf')
    closest_drop = None
    left_to_check = set(drops).difference(found)
    for drop in left_to_check:
        drop_point = (drop.x, drop.y)
        distance = get_distance(drop_point, mid) 
        if distance < closest:
            closest_drop = drop
            closest = distance
    return closest_drop

def update_sections(drops: [Droplet], map_path: Path) -> None:
    '''Periodically Updates the Sections of each droplet to make sure they're in the correct segments'''
    for drop in drops:
        drop.update_section(map_path)
        # print("Current Section: " + str(drop.current_section) + " Droplet (X, Y): (" + str(drop.x) + " , " + str(drop.y) + ")")
    
def update_droplet(closest_droplet: Droplet, mid: (int, int)) -> None:
    '''Updates the Droplets position to the detected box'''
    closest_droplet.x = mid[0]
    closest_droplet.y = mid[1]

def load_windows_files():
    '''Loads the proper files for Windows'''
    model = YOLO("runs\detect\\train10\weights\\best.pt")
    video_cap = cv2.VideoCapture("droplet_videos\\video_data_Rainbow 11-11-22.m4v")
    return model, video_cap

def load_mac_files():
    '''Loads the proper files for Mac'''
    model = YOLO("runs/detect/train10/weights/best.pt")
    video_cap = cv2.VideoCapture("droplet_videos/video_data_Rainbow 11-11-22.m4v")
    return model, video_cap

def build_x_y_map(map_path: Path) -> dict:
    '''Builds a python dictionary that stores every (x, y) coordinate inside a bounding box to map it to a specific queue so we can later check that queue with each detection'''
    ret_dic = {}
    for path in map_path.path:
        x1, y1 = path.top_left
        x2, y2 = path.bottom_right
        for i in range(x1, x2 + 1): 
            for j in range(y1, y2 + 1):
                ret_dic[(i, j)] = path
    return ret_dic

def handle_missings(drops: [Droplet], found: set, map_path: Path) -> None:
    '''This compares the detected droplets vs the actual droplets and then Infers where the missing droplets should be and updates their position that way'''
    missing = set(drops).difference(found)
    for drop in missing:
        drop.update_position(map_path)
    # print("Missing:" + str(missing))

def main():
    drops = []
    map_path = build_path()
    # x_y_map = build_x_y_map(map_path)
    box = sv.BoxAnnotator(text_scale=0.3)

    # map_path.print_paths()

    # Use the right function to your OS and comment out the other
    # model, video_cap = load_windows_files()
    model, video_cap = load_mac_files()

    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return
    
    t = 0 #Temporary Timer for video frames
    droplets_on_screen = 0
    #400 frames is 86.86 seconds with found 92 Seconds for 350 Frames on my PCs
    #Brute Force Mac 87.07 without found 
    #On T == 314, Algorithm will break because It is not handling more than 6 droplets yet
    while t < 300: 
        t += 1

        ret, frame = video_cap.read()
        if not ret:
            print("Video ended")
            break

        if t > 0: #Most of the test functions have to be initialized at 0
            result = model.track(frame, tracker="bytetrack.yaml", persist=True)[0]
            #The following is the core of the Brute Force Logic O(n^2)
            numbers_detected = len(result.boxes.data.tolist())
            droplets_on_screen = get_droplets_on_screen(t, droplets_on_screen, drops) #This will be an assumption that I know shouldn't be a function that's check can just be a counter
            try:
                found = set()
                labels = []
                for data in result.boxes.data.tolist():
                    xone, yone, xtwo, ytwo, _, confidence, _ = data
                    mid = get_mid_point(xone, yone, xtwo, ytwo)
                    closest_droplet = find_closest_droplet(drops, mid, found)
                    found.add(closest_droplet)
                    if confidence:
                        labels.append(f"{closest_droplet.id} {confidence:0.2f}")
                    
                    update_droplet(closest_droplet, mid)

                    if t % 5 == 0: #Every so often update the sections of the droplets does not have to be frequent
                        update_sections(drops, map_path)
                        # print(mid, closest_droplet.x, closest_droplet.y, closest_droplet.id)

                    box_drops(drops, frame)
                    left_mid, right_mid = give_me_a_small_box(mid)
                    cv2.rectangle(frame, left_mid, right_mid, (0, 100, 0), 2)

                if numbers_detected < droplets_on_screen:
                    handle_missings(drops, found, map_path)
            except Exception as e: 
                print(e)
                pass

            detections = sv.Detections.from_ultralytics(result)

            label_path(frame) #This functions draws the bounding boxes on the path
            # where_droplets_should_start(frame)
            label_curves(frame)

            frame = box.annotate(scene=frame, detections=detections, labels = labels)
            cv2.imshow("yolov8", frame)

            if (cv2.waitKey(10) == 27):
                break

def build() -> None:
    '''Just a test replacement function to check so I don't have to run main'''
    map_path = build_path()
    x_y_map = build_x_y_map(map_path)
    print(x_y_map)

if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    # build()