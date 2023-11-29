import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv
import sys, os
import math
import time

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
        self.curve_speed = trajectory

    def update_position(self, course: Path, droplet) -> (int, int):
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
            '''11/29/2023 The following 4 lines commented need to be implement correctly with Straights given the interace if its slanted to a degree.
            I had to comment it out because it conflicts with Straights going Up or Down logic. The rest of the implementation assumes perfect vertical or horizontal straights'''
            # self.x += (self.trajectory * direction_x)
            # #slope = (segment.top_left[1] - segment.top_right[1])/(segment.top_left[0] - segment.top_right[0]) #ideally most  cases slope is 0
            # slope = 0 
            # self.y += slope
            if direction_x and not direction_y:
                self.x += (self.trajectory * direction_x)
            else:
                self.y += (self.trajectory * direction_y)
        else:
            self.x += (self.curve_speed * direction_x)
            self.y = segment.predict_y(self.x)
        self.update_section(course, droplet)
        return (self.x, self.y)
    
    def update_section(self, course: Path, droplet) -> None:
        '''Update which section of the course the droplet is in. 
        In more detail. It generates the constraints in which the coordinate has to be in using the corners of a bounding box. If the detection is
        outside the bounding box we can infer that the droplet moved over to the next section/segment. 
        In this case we remove the droplet from the old section and add it to the new one then carry it over to the one after the new one.
        We do this in order to carry over Droplet data with set logic without running into the error of having 
        calculated too early or too late given the nature in variability of the size of the segments.
        '''
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
        '''
        This function initially intended to calculate trajectory over averages if a Droplet was detected. This then updates over
        the difference between its last difference in straights. The trajectory for curves needs to be decided still.

        11/18/2023 / 8:14 Pm 
        Includes dynamically updating speed based on averages and width of the curves
        Uses width of the curves because the width of the curve projects where it will be in the Y axis. If the curve is narrow and the increments of X
        are too big you'll exit the curve quickly. If it's too wide and increments of X too small then the droplet will move slowly across the curve
        Assumes user passed in thresholds for both straights and curves.

        For a straight the new_trajectory is the average distance traversed over the two detections based on time t

        For a Curve thee average speed is determined by the proximity of center the detection occurred at divided by the total length. 
        Multiplied by the current curve speed and will be updated as long as it's bigger than a threshold. This way it never reaches a too slow of an amount

        #May have to an include an instance of resetting curve speed
        '''
        self.x = mid[0]
        self.y = mid[1]

        if not self.last_detection:
            self.last_detection = (mid, t)
            return
        else:
            if isinstance(x_y_map[mid], Straight):
                try:
                    last_x, curr_x, last_t = self.last_detection[0][0], mid[0], self.last_detection[1]
                    if t != last_t: #This line prevents Zero Division Error
                        new_trajectory =  max((last_x - curr_x), (curr_x - last_x))//max((last_t - t), (t - last_t))
                        if new_trajectory and new_trajectory <= speed_threshold:
                            self.trajectory = new_trajectory
                except AttributeError:
                    print("Attribute Error in Straight")
            else:
                try:
                    current_curve = x_y_map[mid]
                    middle_curve_x = current_curve.mid[0]
                    start_x, end_x = current_curve.start[0], current_curve.end[0]
                    total_length = abs((start_x - end_x))
                    proximity_to_center = abs(middle_curve_x - self.x)
                    if proximity_to_center/total_length * self.curve_speed >= 0.3: 
                        self.curve_speed *= proximity_to_center/total_length
                except:
                    print("Attribute Error in Curve")
            self.last_detection = (mid, t)
                          
class Straight():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        '''Initialize a straight Box and it's direction'''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction
        self.queue = set()
        #self.top_right = (460, 45) # Will have to be a passed in argument once Interface is integrated

    def add_droplet(self, droplet: Droplet) -> None:
        '''Add a droplet to the queue'''
        self.queue.add(droplet)
    
    def remove_droplet(self, droplet: Droplet) -> None:
        '''Removes a droplet from this segments queue'''
        self.queue.remove(droplet)
        
class Curve():
    def __init__(self, point1: (int, int), point2: (int, int), direction: int) -> None:
        '''Initialize a curve's box and it's direction. Assuming a start, middle, end point are provided.
        Initialize a tuple that holds the coefficients to a quadratic formula (a, b, c) for the respective
        f(x) = ax^2 + bx + c
        '''
        self.top_left = point1
        self.bottom_right = point2
        self.direction = direction #pretty confident this variable does not matter since the coeffiecients of quadratics determine directions
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
    
def find_closest_droplet(drops_to_consider: {Droplet}, mid:(int, int)) -> Droplet:
    '''Find the closest droplet to a given (x, y) coordinate provided from a detection. If the droplet was associated already in this round
    skip to save computations'''
    closest = float('inf')
    closest_drop = None
    for drop in drops_to_consider:
        drop_point = (drop.x, drop.y)
        distance = get_distance(drop_point, mid) 
        if distance < closest:
            closest_drop = drop
            closest = distance

    return closest_drop  

def load_mac_files():
    '''Loads the proper files for Mac'''
    model = YOLO("runs/detect/train10/weights/best.pt")
    video_cap = cv2.VideoCapture("droplet_videos/video_data_Rainbow 11-11-22.m4v")
    return model, video_cap

def build_course() -> Path:
    '''This builds the Path object assuming I know the course before hand. Add the segments to the course's queue
    For curves add the start, middle, end points
    
    11/28/2023 This function should be replaced by the interface by drawing out the course. However can be kept to test repeating cases through either hard code or saved file
    '''
    course = Path()

    lst_of_segments = [
    Straight((150, 500), (460, 540), (-1, 0)),  Curve((105, 510), (150, 565), (-1, 1)), Straight((105, 565), (130, 610), (0, 1)),
                    
    Curve((105, 610), (165, 650), (1, 1)), Straight((165, 590), (620, 650), (1, 0)),  Curve((620, 590), (660, 630), (1, 1)),

    Straight((640, 630), (670, 680), (0, 1)), Curve((620, 680), (670, 730), (-1, 1)), Straight((130, 700), (620, 780), (-1, 0)),

    Curve((90, 740), (130, 790), (-1, 1)), Straight((95, 790), (125, 920), (0, 1)), Curve((50, 920), (130, 960), (-1, 1)),

    Straight((40, 680), (80, 920), (0, -1)), Curve((0, 640), (60, 680), (-1, 1)), Straight((0, 680), (25, 920), (0,1)),

    Curve((0, 920), (25, 960), (1, 1))
    ]
    
    lst_of_sme = [None, ((150, 525), (125, 540), (115, 565)), None, ((120, 610), (135, 635), (165, 640)),
                  None, ((620, 610), (640, 615), (650, 630)), None, ((655, 680), (645, 705), (620, 715)), 
                  None, ((130, 755), (112, 762), (105, 785)), None, ((65, 920), (90, 950), (115, 920)), 
                  None, ((0, 680), (25, 650), (50, 680)), None, ((15, 920), (14, 935), (0, 950))
                  ]

    for i in range(len(lst_of_segments)):
        segment = lst_of_segments[i]
        course.add_segment(segment)
        if isinstance(segment, Curve):
            s, m, e = lst_of_sme[i]
            segment.add_sme(s, m, e)
    return course

def label_course(frame, course) -> None:
    '''Draws bounding boxes on Curves for now this is assumed given. 
    This function is just to be used to help visualize the backend can be removed.
    '''
    straight_rgb = (0, 255, 0)
    curve_rgb = (255, 0, 0)
    thick = 2
    for segment in course.segments_in_order:
        cv2.rectangle(frame, segment.top_left, segment.bottom_right, straight_rgb if isinstance(segment, Straight) else curve_rgb, thick)

def label_curves_s_m_e(frame, course) -> None:
    '''Draw the bounding Boxes for the curvers and their Start, Middle, End. 
    Similarly Label Course this can be removed as well and is used
    to label the start middle and end of curves'''
    rgb = (0, 0, 200)
    thick = 2
    for segment in course.segments_in_order:
        if isinstance(segment, Curve):
            start_left, start_right = give_me_a_small_box(segment.start)
            mid_left, mid_right = give_me_a_small_box(segment.mid)
            end_left, end_right = give_me_a_small_box(segment.end)

            cv2.rectangle(frame, start_left, start_right, rgb, thick)
            cv2.rectangle(frame, mid_left, mid_right, rgb, thick)
            cv2.rectangle(frame, end_left, end_right, rgb, thick)

def get_droplets_on_screen(t : int, num_droplets: int, drops:{Droplet}, course) -> int:
    '''Initializes Droplet objects this is assumed I know it before hand T == Frame they appear in
    11/29/2023. Can simplify this function by taking a list time T and list of Dispensers to initialize thes droplets and make it a for loop
    Note that this function does exist inside another loop so .add() may indefinitely add it. One way to do it is check the length of it and to not add if it is
    or simply condense this function and move it outside of the for loop.
    '''
    
    #(325, 510), (335, 520)
    if t == 41:
        #(330, 515)
        droplet_1 = Droplet(1, 330, 515, 5)
        course.add_droplet_to_queues(droplet_1)
        drops.add(droplet_1)
        return 1
    elif t == 51:
        droplet_2 = Droplet(2, 330, 515, 5)
        course.add_droplet_to_queues(droplet_2)
        drops.add(droplet_2)
        return 2
    elif t == 61:
        droplet_3 = Droplet(3, 330, 515, 5)
        course.add_droplet_to_queues(droplet_3)
        drops.add(droplet_3)
        return 3
    elif t == 69:
        droplet_4 = Droplet(4, 330, 515, 5)
        course.add_droplet_to_queues(droplet_4)
        drops.add(droplet_4)
        return 4
    elif t == 77:
        droplet_5 = Droplet(5, 330, 515, 5)
        course.add_droplet_to_queues(droplet_5)
        drops.add(droplet_5)
        return 5
    elif t == 86:
        droplet_6 = Droplet(6, 330, 515, 5)
        course.add_droplet_to_queues(droplet_6)
        drops.add(droplet_6)
        return 6
    elif t == 95:
        droplet_7 = Droplet(7, 330, 515, 5)
        course.add_droplet_to_queues(droplet_7)
        drops.add(droplet_7)
        return 7
    elif t == 103:
        droplet_8 = Droplet(8, 330, 515, 5)
        course.add_droplet_to_queues(droplet_8)
        drops.add(droplet_8)
        return 8 
    elif t == 110:
        droplet_9 = Droplet(9, 330, 515, 5)
        course.add_droplet_to_queues(droplet_9)
        drops.add(droplet_9)
        return 9
    elif t == 120:
        droplet_10 = Droplet(10, 330, 515, 5)
        course.add_droplet_to_queues(droplet_10)
        drops.add(droplet_10)
        return 10
    elif t == 129:
        droplet_11 = Droplet(11, 330, 515, 5)
        course.add_droplet_to_queues(droplet_11)
        drops.add(droplet_11)
        return 11
    else:
        return num_droplets

def where_droplets_should_start(frame) -> None:
    '''Draws a bounding box in front of dispenser location'''
    cv2.rectangle(frame, (325, 510), (335, 520), (255, 0, 0), 2)
    
def build_x_y_map(course: Path) -> {(int, int): Path}:
    '''
    Builds a python dictionary that stores every (x, y) coordinate inside a path segment/section
    to map it to a specific segment so we can later check that queue associated to that segment 
    with each detection
    '''
    ret_dic = {}
    for course in course.segments_in_order:
        x1, y1 = course.top_left
        x2, y2 = course.bottom_right
        smaller_x, bigger_x = min(x1, x2), max(x1, x2)
        smaller_y, bigger_y = min(y1, y2), max(y1, y2)
        for i in range(smaller_x, bigger_x + 1): 
            for j in range(smaller_y, bigger_y + 1):
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

def label_droplets(drops: {Droplet}, frame) -> None:
    '''This boxs the Droplets I know about'''
    for drop in drops:
        left_predict, right_predict = give_me_a_small_box((drop.x, drop.y))
        cv2.rectangle(frame, left_predict, right_predict, (255, 255, 0), 4)

def handle_missings(drops: {Droplet}, found: set, map_course: Path) -> None:
    '''This compares the detected droplets vs the actual droplets and then Infers where the missing droplets should be and updates their position that way'''
    missing = drops.difference(found)
    for drop in missing:
        drop.update_position(map_course, drop)
        found.add(drop)

def load_mac_files():
    '''Loads the proper files for Mac'''
    model = YOLO("runs/detect/train10/weights/best.pt")
    video_cap = cv2.VideoCapture("droplet_videos/video_data_Rainbow 11-11-22.m4v")
    return model, video_cap 

def main(weights_path, video_path):
    '''Initializes all the variables the set of all droplets to help check for the missing droplets. The course that holds all the segments on Straights and Curves
    x_y_map = {(x, y) = Segment} is a dictionary that maps all x,y points in side of each segment to that particular segment. Allows for looking up Droplets in that section
    speed_threshold prevents detection from increasing average speed to beyond a reasonable speed.
    '''
    all_droplets = set()
    course = build_course()
    x_y_map = build_x_y_map(course)
    box = sv.BoxAnnotator(text_scale=0.3)
    speed_threshold = 5
    # model, video_cap = load_mac_files()
    model = YOLO(weights_path)
    video_cap = cv2.VideoCapture(video_path)

    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    '''Initializes Time t to help debug on specific frames or time intervals of the video. Droplets on screen is a counter for how many droplets to expect at any given time t'''    
    t = 0
    droplets_on_screen = 0
    # while t < 500: 
    while video_cap.isOpened():
        t += 1 #Increment the time

        '''Open the video frames and play it'''
        ret, frame = video_cap.read()

        frame = cv2.resize(frame, (1280, 1024))

        if not ret:
            print("Video ended")
            break

        if t > 0:
            print(t)
            '''Droplets on screen is to get how many droplets should be on screen at any given time t.
            Result holds the model's detections. Found set is used to be compared to all the droplets to see if there's a mising one. Numbers detected operates similarly
            Labels is to generate the strings/text for the bounding boxes of the detections
            '''
            droplets_on_screen = get_droplets_on_screen(t, droplets_on_screen, all_droplets, course)
            result = model.track(frame, tracker="bytetrack.yaml", persist=True)[0]
            numbers_detected = len(result)
            found = set()
            labels = []
            
            try:
                '''Data is from the models detection in the formate of top left point, bottom right point, __, confidence, class from the data set
                mid is the middle point of two points. used in this case for the top left and bottom right point of each detection
                '''
                for data in result.boxes.data.tolist():
                    try:
                        if len(data) == 7:
                            xone, yone, xtwo, ytwo, id, confidence, class_in_model = data
                        else:
                            xone, yone, xtwo, ytwo, confidence, class_in_model = data
                    except ValueError:
                        if not data:
                            print("No Data given by detection")
                        else:
                            print("Error occurred while unpacking data provided from model")

                    mid = get_mid_point(xone, yone, xtwo, ytwo)

                    '''The following try except clause is used for debugging or handling edge cases.'''
                    try:
                        '''drops to consider is ideally always the drops in the segment closest to the detection'''
                        drops_to_consider = x_y_map[mid].queue
                        # drops_to_consider = all_droplets

                    except KeyError:
                        '''A Key Error occurs when a detection happens outside of the Course in space that should not be considered.
                        Will skip any computation for consideration and flag the false detection. Should be a True False occurrence
                        '''
                        print("Detection occurred outside of the course. Data: ", data)
                        continue
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        continue
                        
                    '''Find Closest Droplet takes a set of drops in the segment and compares it to the detections then adds it to find'''
                    closest_droplet = find_closest_droplet(drops_to_consider, mid)
                    if not closest_droplet:
                        continue
                    found.add(closest_droplet)

                    closest_droplet.update_last_seen(mid, t, x_y_map, speed_threshold)
                    # closest_droplet.update_position(course)
                        
                    '''-------------------------------------------------------------'''

                    #Error could be here as well since data is stored in two segments at a time
                    '''If the current section that the detection was found in isn't the same registered with the droplet
                    update the droplet's current position in that section. This results in removing itself in the previous segment. Making sure it's
                    in the section it was discovered in as well as carrying the information over the to the next section'''
                    if x_y_map[mid] != course.segments_in_order[closest_droplet.current_section]:
                        closest_droplet.update_section(course, closest_droplet)
                    
                    '''The remainder of the code is the labeling and drawing of the map on the frame'''
                    label_droplets(all_droplets, frame)

                    if confidence:
                        labels.append(f"{closest_droplet.id} {confidence:0.2f}")

                if numbers_detected < droplets_on_screen:
                    print("Handling Missing Cases")
                    handle_missings(all_droplets, found, course)

            except Exception as e:
                print(type(closest_droplet))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

        # where_droplets_should_start(frame)  #Call to show dispenser locations

        detections = sv.Detections.from_ultralytics(result)
        label_course(frame, course) 
        label_curves_s_m_e(frame, course)

        frame = box.annotate(scene=frame, detections=detections, labels = labels)
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(10) == 27):
            break

if __name__ == '__main__':
    '''Start Time and End Time is a timer to measure run time'''
    start_time = time.perf_counter()
    # main("runs/detect/train10/weights/best.pt", "droplet_videos/video_data_Rainbow 11-11-22.m4v")
    # main("runs/detect/train3/weights/best.pt", "droplet_videos/1_onedroplet_raw.mp4")
    main("runs/detect/train3/weights/best.pt", "droplet_videos/6_smalldropletsfast_raw.mp4")
    # main("runs/detect/best.pt", "droplet_videos/6_smalldropletsfast_raw.mp4")
    # build() #Just a test function to isolate portions
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")