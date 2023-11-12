import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv
import sys, os
import math
import time

class Path():
    def __init__(self):
        self.segments_in_order = []
        self.length = 0
    
    def add_segment(self, new_segment):
        self.segments_in_order.append(new_segment)
        self.length += 1

    def add_droplet_to_queues(self, droplet):
        '''Adds droplets to the corresponding queue in each segment. If the segment isn't last in the list then
        add it to the correct one and the next one. That way we can infer will it will be and remove it accordingly'''
        if len(self.segments_in_order) > 1 and droplet.current_section + 1 < self.length:
            self.segments_in_order[droplet.current_section + 1].add_droplet(droplet)
        self.segments_in_order[droplet.current_section].add_droplet(droplet)

class Droplet():
    def __init__(self, id, x: int = None, y:int = None, trajectory: int = 1, current_section: int = 0) -> None:
        '''Initialize Droplet Object'''
        self.id = id
        self.x = x
        self.y = y
        self.trajectory = trajectory
        self.current_section = current_section

    def update_position(self, course: Path) -> (int, int):
        '''Update a droplets position using the assumption of what direction it's traveling in from it's corresponding course.'''
        segment = course.segments_in_order[self.current_section]
        direction_x, direction_y = segment.direction
        self.x += (self.trajectory * direction_x)
        self.y += (self.trajectory * direction_y)
        return (self.x, self.y)
    
    def update_section(self, course: Path, droplet) -> None:
        '''Update which section of the course the droplet is in'''
        segment = course.segments_in_order[self.current_section]
        left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
        if self.x < left or self.x > right or self.y < top or self.y > bot:
            if self.current_section < len(course.segments_in_order):
                course.segments_in_order[self.current_section].remove_droplet(droplet)
                self.current_section += 1
                course.segments_in_order[self.current_section].add_droplet(droplet)
                if self.current_section + 1 < course.length:
                    course.segments_in_order[self.current_section + 1].add_droplet(droplet)



class Straight():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        '''Initialize a straight Box and it's direction'''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction
        self.queue = set()

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
        self.start = None
        self.mid = None
        self.end = None
        self.queue = set()

    def add_droplet(self, droplet: Droplet):
        self.queue.add(droplet)
    
    def remove_droplet(self, droplet: Droplet):
        self.queue.remove(droplet)

def load_mac_files():
    '''Loads the proper files for Mac'''
    model = YOLO("runs/detect/train10/weights/best.pt")
    video_cap = cv2.VideoCapture("droplet_videos/video_data_Rainbow 11-11-22.m4v")
    return model, video_cap

def build_course() -> Path:
    '''This builds the Path object assuming I know the course before hand'''
    course = Path()

    straight1 = Straight((75, 50), (460, 70), (-1, 0)) #Left
    course.add_segment(straight1)

    curve1 = Curve((25, 50), (75, 110), (-1, 1)) #Left and Down
    course.add_segment(curve1)

    straight2 = Straight((45, 110), (60, 160), (0, 1)) #Down
    course.add_segment(straight2)

    curve2 = Curve((40, 160), (100, 205), (1, 1)) #Right and Down
    course.add_segment(curve2)

    straight3 = Straight((100, 180), (530, 205), (1, 0))
    course.add_segment(straight3)

    return course

def build_x_y_map(course: Path) -> dict:
    '''Builds a python dictionary that stores every (x, y) coordinate inside a bounding box to map it to a specific queue so we can later check that queue with each detection'''
    ret_dic = {}
    for course in course.segments_in_order:
        x1, y1 = course.top_left
        x2, y2 = course.bottom_right
        for i in range(x1, x2 + 1): 
            for j in range(y1, y2 + 1):
                ret_dic[(i, j)] = course
    return ret_dic

def get_distance(point1: (int, int), point2: (int, int)) -> float:
    '''Distance formula between two points'''
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)** 2 + (y2 - y1) ** 2)

def get_mid_point(xone: int, yone: int, xtwo: int, ytwo: int) -> (int, int):
    '''Take two corners and return the middle of the two points'''
    return ((xone + xtwo)//2, (yone + ytwo)//2)

def give_me_a_small_box(point: (int, int)) -> ((int, int), (int, int)):
    '''Creates a small box for open CV to generate a bounding box a round a point'''
    #Open CV doesn't support float objects
    return (int(point[0] - 2), int(point[1] - 2)),(int(point[0] + 2), int(point[1] + 2))

def update_droplet(closest_droplet: Droplet, mid: (int, int)) -> None:
    '''Updates the Droplets position to the detected box'''
    closest_droplet.x = mid[0]
    closest_droplet.y = mid[1]

def label_course(frame) -> None:
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

def get_droplets_on_screen(t : int, num_droplets: int, drops:{Droplet}, course) -> int:
    '''Initializes Droplet objects this is assumed I know it before hand T == Frame they appear in'''
    if t == 1:
        #(450.0, 60.0) 3
        droplet_1 = Droplet(1, 450, 60, 2)
        course.add_droplet_to_queues(droplet_1)
        drops.add(droplet_1)
        return 1
    elif t == 114:
        #(316.0, 62.0) 114
        droplet_2 = Droplet(2, 315, 60, 1)
        course.add_droplet_to_queues(droplet_2)
        drops.add(droplet_2)
        return 2
    elif t == 147:
        #(316.0, 62.0) 114
        droplet_3 = Droplet(3, 315, 60, 1)
        course.add_droplet_to_queues(droplet_3)
        drops.add(droplet_3)
        return 3
    elif t == 152:
        #(449.0, 60.0) 148
        droplet_4 = Droplet(4, 450, 60, 1)
        course.add_droplet_to_queues(droplet_4)
        drops.add(droplet_4)
        return 4
    elif t == 185:
        #(448.0, 60.0) 184
        droplet_5 = Droplet(5, 450, 60, .5)
        course.add_droplet_to_queues(droplet_5)
        drops.add(droplet_5)
        return 5
    elif t == 222:
        #(449.0, 60.0) 148
        droplet_6 = Droplet(6, 450, 60, .5)
        course.add_droplet_to_queues(droplet_6)
        drops.add(droplet_6)
        return 6
    else:
        return num_droplets
def find_closest_droplet(drops_to_consider: {Droplet}, mid:(int, int), found: set) -> Droplet:
    closest = float('inf')
    closest_drop = None
    for drop in drops_to_consider:
        if drop in found:
            print("Skipping")
            continue
        drop_point = (drop.x, drop.y)
        distance = get_distance(drop_point, mid) 
        if distance < closest:
            closest_drop = drop
            closest = distance
    return closest_drop  

def box_drops(drops: {Droplet}, frame) -> None:
    '''This boxs the Droplets I know about'''
    for drop in drops:
        left_predict, right_predict = give_me_a_small_box((drop.x, drop.y))
        cv2.rectangle(frame, left_predict, right_predict, (100, 0, 0), 4)

def handle_missings(drops: {Droplet}, found: set, map_course: Path) -> None:
    '''This compares the detected droplets vs the actual droplets and then Infers where the missing droplets should be and updates their position that way'''
    missing = drops.difference(found)
    for drop in missing:
        drop.update_position(map_course)
        found.add(drop)

def main():
    #the moment the closest droplet's .current_section 
    #is different from the course from x_y_map update sections first then do calcs
    all_droplets = set()
    course = build_course()
    x_y_map = build_x_y_map(course)
    box = sv.BoxAnnotator(text_scale=0.3)
    model, video_cap = load_mac_files()

    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return
    
    t = 0
    droplets_on_screen = 0
    while t < 300: 
            t += 1

            ret, frame = video_cap.read()
            if not ret:
                print("Video ended")
                break
    
            if t > 0:
                droplets_on_screen = get_droplets_on_screen(t, droplets_on_screen, all_droplets, course)
                result = model.track(frame, tracker="bytetrack.yaml", persist=True)[0]
                numbers_detected = 0
                found = set()
                labels = []
                try:
                    for data in result.boxes.data.tolist():
                        xone, yone, xtwo, ytwo, _, confidence, _ = data
                        mid = get_mid_point(xone, yone, xtwo, ytwo)
                        drops_to_consider = x_y_map[mid].queue
                        closest_droplet = find_closest_droplet(drops_to_consider, mid, found)
                        numbers_detected += 1
                        found.add(closest_droplet)
                        update_droplet(closest_droplet, mid) 

                        if x_y_map[mid] != course.segments_in_order[closest_droplet.current_section]:
                            closest_droplet.update_section(course, closest_droplet)
                            print("New Closest Droplet Section: ")
                            print(closest_droplet.current_section)
                        box_drops(all_droplets, frame)

                        if confidence:
                            labels.append(f"{closest_droplet.id} {confidence:0.2f}")

                    if numbers_detected < droplets_on_screen:
                        handle_missings(all_droplets, found, course)

                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

            detections = sv.Detections.from_ultralytics(result)

            label_course(frame) 
            label_curves(frame)

            frame = box.annotate(scene=frame, detections=detections, labels = labels)
            cv2.imshow("yolov8", frame)

            if (cv2.waitKey(10) == 27):
                break


if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
