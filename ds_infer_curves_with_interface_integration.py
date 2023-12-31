import cv2
from ultralytics import YOLO
import supervision as sv
import sys, os
import math


class Path():
    def __init__(self) -> None:
        self.segments_in_order = []
        # added to story segments

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

class Droplet():
    def __init__(self, id, x: int = None, y:int = None, trajectory: int = 1, current_section: int = 0) -> None:
        '''Initialize Droplet Object'''
        self.id = id
        self.x = x
        self.y = y
        self.trajectory = trajectory
        self.current_section = current_section
        self.last_detection = None

    def update_position(self, course: Path) -> (int, int):
        '''Update a droplets position using the assumption of what direction it's traveling in from it's corresponding course.
        If the droplet is in a Straight then the case is update the direction depending on a flat trajectory. 
        If it's on a curve then use the assumption we know the start, middle, and end point. Calculate the Coefficients of a Quadratic Equation given three points.
        This assumes that all curves are Quadratic in nature and has a start, middle, end. More information for the quadratic process is in the coming functions

         11/18/2023:
        self.curve speed should be dynamically changed, initialize at initial droplet trajectory.
        When Interface is Integrated replace the commented slope variable
        In concept Direction y shouldn't matter because slope is calculated from top left and top right
        where left < right and will divide into
        Will be accordingly negative or positive.
        May have a Zero Division Error if a Segment box is ever one pixel which should never happen. But if that ever wanted to be handled, it can be done
        by adding a conditional if segment.top_left[0] != segment.top_right[0]
        '''
        segment = course.segments_in_order[self.current_section]
        direction_x, direction_y = segment.direction
        if isinstance(segment, Straight):
            self.x += (self.trajectory * direction_x)
            # slope = (segment.top_left[1] - segment.top_right[1])/(segment.top_right[0] - segment.top_left[0]) #ideally most  cases slope is 0
            slope = 0 # TODO When Interface is Integrated replace this with above ^^^
            self.y += slope
        else:
            try:
                try:
                    self.x += (self.curve_speed * direction_x)
                except AttributeError:
                    print("Occured o nself.x")
                self.y = segment.predict_y(self.x)
            except AttributeError:
                print("Occurred here")
        return (self.x, self.y)
    
    def update_section(self, course: Path, droplet) -> None:
        '''Update which section of the course the droplet is in. 
        In more detailed generate the constraints in which the coordinate has to be in using the corners of a bounding box. if outside the bounding box
        we can infer that the droplet moved over to the next section. Remove the droplet from the old section. Add it to the new one and carry it over to the one after the new one
        We do this in order to carry over Droplet data with set logic without running into the error of having calculated too early or too late given the nature in variability of the size of the segments.
        '''
        segment = course.segments_in_order[self.current_section]
        left, right, top, bot = segment.top_left[0], segment.bottom_right[0], segment.top_left[1], segment.bottom_right[1]
        if self.x < left or self.x > right or self.y < top or self.y > bot:
            if self.current_section < len(course.segments_in_order):
                course.segments_in_order[self.current_section].remove_droplet(droplet)
                self.current_section += 1
                course.segments_in_order[self.current_section].add_droplet(droplet)
                if self.current_section + 1 < len(course.segments_in_order):
                    course.segments_in_order[self.current_section + 1].add_droplet(droplet)
    
    def update_last_seen(self, mid : (int, int), t : int, x_y_map: {(int, int): Path}) -> None:
        '''In Progress 11/13/2023 1:07 AM Something with this is breaking occassionally will have to see what
        This function initially intended to calculate trajectory over averages if a Droplet was detected. This then updates over
        the difference between its last difference in straights. The trajectory for curves needs to be decided still.
        '''
        self.x = mid[0]
        self.y = mid[1]
        if not self.last_detection:
            self.last_detection = (mid, t)
            return
        else:
            if isinstance(x_y_map[mid], Straight):
                last_x, curr_x, last_t = self.last_detection[0][0], mid[0], self.last_detection[1]
                new_trajectory =  max((last_x - curr_x), (curr_x - last_x))//max((last_t - t), (t - last_t))
                if new_trajectory:
                    self.trajectory = new_trajectory
                self.last_detection = (mid, t)
                # print("Droplet ID: " + str(self.id) + " New trajectory: " + str(self.trajectory))
            # else:
            #     #WILL PROBABLY HAVE TO UPDATE THE TRAJECTORY TO BE THE AVERAGE OF A NEIGHBORING SEGMENT OR AL LDROPLETS
            #     current_curve = x_y_map[mid]
            #     print(current_curve.predict_y(mid[0]))

class Straight():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        '''Initialize a straight Box and it's direction'''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction
        self.queue = set()

    def add_droplet(self, droplet: Droplet) -> None:
        '''Add a droplet to the queue'''
        self.queue.add(droplet)
    
    def remove_droplet(self, droplet: Droplet) -> None:
        self.queue.remove(droplet)
        
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
    

def load_mac_files(file):
    '''Loads the proper files for Mac'''
    model = YOLO("best.pt")
    # TODO add way to pull up video with file explorer maybe have it done in interface
    video_cap = cv2.VideoCapture(file)
    # video_cap = cv2.VideoCapture("droplet_videos/5_smalldroplets_raw.mp4")
    return model, video_cap


def build_course(bound_list) -> Path:
    '''This builds the Path object assuming I know the course before hand. Add the segments to the course's queue
    For curves add the start, middle, end points'''
    course = Path()

    for box in bound_list:
        new_box = ""
        print(box)
        if box[1] == "straight":
            new_box = Straight((box[3][0]), (box[3][1]), box[2])
            course.add_segment(new_box)
        else:
            new_box = Curve((box[3][0]), (box[3][1]), box[2])
            new_box.add_sme(box[5][0], box[4], box[5][1])
            course.add_segment(new_box)


    # straight1 = Straight((75, 50), (460, 70), (-1, 0)) #Left
    # course.add_segment(straight1)
    #
    # curve1 = Curve((25, 50), (75, 110), (-1, 1)) #Left and Down
    # curve1.add_sme((75, 60), (60, 80), (52, 110))
    # course.add_segment(curve1)
    #
    # straight2 = Straight((45, 110), (60, 160), (0, 1)) #Down
    # course.add_segment(straight2)
    #
    # curve2 = Curve((40, 160), (100, 205), (1, 1)) #Right and Down
    # curve2.add_sme((50, 160), (70, 190), (100, 195))
    # course.add_segment(curve2)
    #
    # straight3 = Straight((100, 180), (530, 205), (1, 0))
    # course.add_segment(straight3)
    return course

def build_x_y_map(course: Path) -> {(int, int): Path}:
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

def label_course(frame, bound_box_list) -> None:
    '''Draws bounding boxes on Curves for now this is assumed given'''
    # [boundingNum, "straight"|"curved", direction, [(initial x, initial y), (x,y)], OPTIONAL (x, y)]
    'TODO: loop through and add visual for bounding boxes '
    for box in bound_box_list:
        cv2.rectangle(frame, box[3][0], box[3][1], (0, 255, 0), 2)
def label_curves_s_m_e(frame, bound_box_list) -> None:
    '''Draw the bounding Boxes for the curvers and their Start, Middle, End'''
    'TODO: loop and add bounding boxes '
    for box in bound_box_list:
        if box[1] == "curved":
            cv2.rectangle(frame, box[3][0], box[3][1], (0, 255, 0), 2)
            give_me_a_small_box(box[4])

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
    # elif t == 250:
    #     #(449.0, 60.0) 148
    #     droplet_7 = Droplet(7, 450, 60, 4)
    #     course.add_droplet_to_queues(droplet_7)
    #     drops.add(droplet_7)
    #     return 7
    else:
        return num_droplets

def find_closest_droplet(drops_to_consider: {Droplet}, mid:(int, int), found: set) -> Droplet:
    '''Find the closest droplet to a given (x, y) coordinate provided from a detection. If the droplet was associated already in this round
    skip to save computations'''
    closest = float('inf')
    closest_drop = None
    for drop in drops_to_consider.difference(found):
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


def print_path_segments(course: Path, t: int) -> None:
    '''Debug Function that prints T frames and segment the droplets in each segment and shows them updating'''
    print(t)
    for segment in course.segments_in_order:
        print([drop.id for drop in segment.queue])
    print("\n")

def main(bound_box_list):
    all_droplets = set()
    # call bounding function hereEEE

    model, video_cap = load_mac_files()

    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return


    course = build_course(bound_box_list) # TODO fix
    x_y_map = build_x_y_map(course)
    box = sv.BoxAnnotator(text_scale=0.3)
    
    t = 0
    droplets_on_screen = 0
    while t < 350: 
            t += 1

            ret, frame = video_cap.read()
            frame = cv2.resize(frame, (640, 480)) # default frame size, reduce bounding boxes to match


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

                        closest_droplet.update_last_seen(mid, t, x_y_map)
                        # closest_droplet.update_position(course)

                        if x_y_map[mid] != course.segments_in_order[closest_droplet.current_section]:
                            closest_droplet.update_section(course, closest_droplet)
                        
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
            label_course(frame, bound_box_list)
            label_curves_s_m_e(frame,bound_box_list)

            frame = box.annotate(scene=frame, detections=detections, labels = labels)
            cv2.imshow("yolov8", frame)

            if (cv2.waitKey(10) == 27):
                break

