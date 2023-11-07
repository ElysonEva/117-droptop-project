import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv
import os
import math
from random import randint

class Droplet():
    def __init__(self, id, x = None, y = None, trajectory = 1, current_section = 0):
        '''Initialize Droplet Object'''
        self.id = id
        self.x = x
        self.y = y
        self.trajectory = trajectory
        self.current_section = current_section

    def update_position(self, path):
        '''Update it's position with assumptions of direction and current position'''
        segment = path.path[self.current_section]
        direction_x, direction_y = segment.direction
        self.x += (self.trajectory * direction_x)
        self.y += (self.trajectory * direction_y)
        return (self.x, self.y)
    
    def update_section(self, path):
        '''Update which section of the path the droplet is in'''
        segment = path.path[self.current_section]
        left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
        print("Pos: ", left, right, top, bot)
        if self.x < left or self.x > right or self.y < top or self.y > bot:
            if self.current_section < len(path.path):
                self.current_section += 1

    # def update_trajectory(self):

        
class Path():
    def __init__(self):
        self.path = []

    def add_to_path(self, new_section):
        '''Add a path to the list of paths in order'''
        self.path.append(new_section)

    def print_paths(self):
        '''Print the paths'''
        for path in self.path:
            print(path.top_left, path.bottom_right, path.direction)


class Straight():
    def __init__(self, point1, point2, direction):
        '''Initialize a straight Box and it's direction'''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction

class Curve():
    def __init__(self, point1, point2, direction):
        '''Initialize a curve's box and it's direction'''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction

def check_file(path):
    if os.path.isfile(path):
        print("Works")
    else:
        print("Failed")

def load_data():
    rf = Roboflow(api_key="Izum4d9L9gLJ14xMz3sx")
    project = rf.workspace("dropshop-froos").project("dropshop")
    dataset = project.version(1).download("yolov8")

def get_mid_point(xone, yone, xtwo, ytwo):
    return ((xone + xtwo)//2, (yone + ytwo)//2)

def give_me_a_small_box(point):
    #Open CV doesn't support float objects
    return (int(point[0] - 2), int(point[1] - 2)),(int(point[0] + 2), int(point[1] + 2))

def label_path(frame):
    '''Draws bounding boxes on Curves for now this is assumed given'''
    cv2.rectangle(frame, (75, 50), (460, 70), (0, 255, 0), 2) #First Straight
    cv2.rectangle(frame, (25, 50), (75, 110), (0, 200, 0), 2) #First Curve
    cv2.rectangle(frame, (45, 110), (60, 160), (0, 255, 0), 2) #Second Straight from First Curve to Second Curve # First Vertical
    cv2.rectangle(frame, (40, 160), (100, 205), (0, 200, 0), 2) #Second Curve
    cv2.rectangle(frame, (100, 180), (530, 205), (0, 255, 0), 2) #Third Straight

def where_droplets_should_start(frame):
    '''Draws a bounding box in front of dispenser location'''
    cv2.rectangle(frame, (445, 55), (455, 65), (255, 0, 0), 2) #Droplet 1, 4, 5, 6
    cv2.rectangle(frame, (315, 55), (325, 65), (255, 0, 0), 2) #Droplet 2, 3

def get_droplets_on_screen(t, num_droplets, drops):
    '''Initializes Droplet objects this is assumed I know it before hand'''
    #Droplet 1 T == 1
    #Droplet 2 T == 113
    #Droplet 3 T == 147
    #Droplet 4 T == 152
    #Droplet 5 T == 185
    #Droplet 6 T == 222
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

def build_path():
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

def box_drops(drops, frame):
    '''This boxs the Droplets I know about'''
    for drop in drops:
        left_predict, right_predict = give_me_a_small_box((drop.x, drop.y))
        cv2.rectangle(frame, left_predict, right_predict, (100, 0, 0), 4)
        print(drop.x, drop.y, drop.id, drop.current_section)

def get_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)** 2 + (y2 - y1) ** 2)

def find_closest_droplet(drops, mid):
    closest = float('inf')
    closest_drop = None
    for i in range(len(drops)):
        drop = drops[i]
        drop_point = (drop.x, drop.y)
        distance = get_distance(drop_point, mid) 
        if distance < closest:
            closest_drop = i
            closest = distance
    return drops[closest_drop]

def update_sections(drops, map_path):
    '''Periodically Updates the Sections of each droplet to make sure they're in the correct segments'''
    for drop in drops:
        drop.update_section(map_path)
        print("Current Section: " + str(drop.current_section) + " Droplet (X, Y): (" + str(drop.x) + " , " + str(drop.y) + ")")
    
def update_droplet(closest_droplet, mid):
    '''Updates the Droplets position to the detected box'''
    closest_droplet.x = mid[0]
    closest_droplet.y = mid[1]

def build_labels(data, drops, labels):
    '''Builds  the labels into a string for the Model to be labeled'''
    label_mid = get_mid_point(data[0], data[1], data[2], data[3])
    closest_id = str(find_closest_droplet(drops, label_mid).id)
    labels.append(f"{closest_id} {data[5]:0.2f}")
    return

def load_windows_files():
    model = YOLO("runs\detect\\train10\weights\\best.pt")
    video_cap = cv2.VideoCapture("droplet_videos\\video_data_Rainbow 11-11-22.m4v")
    return model, video_cap

def load_mac_files():
    model = YOLO("runs/detect/train10/weights/best.pt")
    video_cap = cv2.VideoCapture("droplet_videos/video_data_Rainbow 11-11-22.m4v")
    return model, video_cap
def main():
    drops = []
    map_path = build_path()
    box = sv.BoxAnnotator(text_scale=0.3)
    # map_path.print_paths()

    # model, video_cap = load_windows_files()
    model, video_cap = load_mac_files()

    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return
    
    t = 0 #Temporary Timer for video frames
    droplets_on_screen = 0
    while t < 400:
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
                    build_labels(data, drops, labels)

                    xone, yone, xtwo, ytwo, _, _, _ = data
                    mid = get_mid_point(xone, yone, xtwo, ytwo)
                    #I have every detected in mid, I have the total droplets_on_screen,
                    closest_droplet = find_closest_droplet(drops, mid)
                    update_droplet(closest_droplet, mid)
                    if numbers_detected < droplets_on_screen:
                        found.add(closest_droplet)
                        
                    if t % 5 == 0: #Every so often update the sections of the droplets does not have to be frequent
                        update_sections(drops, map_path)
                        # print(mid, closest_droplet.x, closest_droplet.y, closest_droplet.id)
                    box_drops(drops,  frame)
                    left_mid, right_mid = give_me_a_small_box(mid)
                    cv2.rectangle(frame, left_mid, right_mid, (0, 100, 0), 2)
                if numbers_detected < droplets_on_screen:
                    missing = set(drops).difference(found)
                    for drop in missing:
                        drop.update_position(map_path)
                    # print("Missing:" + str(missing))
            except Exception as e: 
                print(e)
                pass

            detections = sv.Detections.from_ultralytics(result)

            label_path(frame) #This functions draws the bounding boxes on the path
            # where_droplets_should_start(frame)

            frame = box.annotate(scene=frame, detections=detections, labels = labels)
            cv2.imshow("yolov8", frame)

            if (cv2.waitKey(10) == 27):
                break

if __name__ == '__main__':
    main()