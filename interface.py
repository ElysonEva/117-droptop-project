import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
# TODO: move to interface class later
file_path = None
# when deleting box: delete in bounding_boxes, relabel bounding boxes from the position of the delete box
# Todo load/save bounding boxes for images -> edit mode
#
bounding_boxes = []
bounding_num = 1
# Bounding box number

createBox = True
drawing = False
ix, iy, x, y, = None, None, None, None
cap = None
value = 1
direction = "Up"
delete_mode = 1

root = tk.Tk()
root.title('Bounding box creation: Please Select bounding boxes')


"""
    sets the direction of the bounding box
    Updates the text for the TK inter menu 
"""
def set_direction(value):
    global direction
    direction = value
    direction_label.config(text=f"Selected Direction: {direction}")

"""
    Toggle function that changes if the current bounding box is a 'straight' or 'curve'
    Updates the text for the TK inter menu 
"""
def toggle_value():
    # 1 == straight, -1 curve
    global value
    value = -value
    typeVal = "Straight"
    if value != 1: typeVal = "Curve"
    bounding_label.config(text=f"Bounding Box Type: {typeVal}")

"""
    Toggle function that changes if the user is creating or deleting bounding box
"""
def toggle_delete():
    global delete_mode
    delete_mode = -delete_mode
    typeVal = "Create"
    if delete_mode != 1: typeVal = "Delete"
    delete_label.config(text=f"Mode: {typeVal}")

def draw_frame_with_boxes():
    temp_frame = frame.copy()
    for box in bounding_boxes:
        box[0] = bounding_boxes.index(box)
        x1, y1 = box[3][0][0], box[3][0][1]
        x2, y2 = box[3][1][0], box[3][1][1]
        cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(temp_frame, str(box[0]), ((x1 + x2) // 2, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Add number label

    return temp_frame


def open_file():
    global createBox, cap, direction
    global file_path
    global frame
    file_path = filedialog.askopenfilename() # opens files to get video
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        print("It's a video")
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1440, 720)) # frame zoomed in without readjustment
        cv2.imshow('First Frame', frame)
        cv2.setMouseCallback('First Frame', draw_box)  # Assuming the draw_box function is defined somewhere in your code


        direction_menu = tk.Menu(root)
        root.config(menu=direction_menu)

        direction_submenu = tk.Menu(direction_menu, tearoff=0)
        direction_menu.add_cascade(label="Direction", menu=direction_submenu)
        direction_menu.add_command(label="Up", command=lambda: set_direction("Up"))
        direction_menu.add_command(label="Down", command=lambda: set_direction("Down"))
        direction_menu.add_command(label="Left", command=lambda: set_direction("Left"))
        direction_menu.add_command(label="Right", command=lambda: set_direction("Right"))

        direction_menu.add_command(label="Toggle Type", command=toggle_value)
        direction_menu.add_command(label="Toggle Mode", command=toggle_delete)

        global direction_label, bounding_label, delete_label
        delete_label = tk.Label(root, text=f"Mode: Create ", padx=10, pady=10)
        delete_label.pack()

        direction_label = tk.Label(root, text=f"Selected Direction: {direction}", padx=10, pady=10)
        direction_label.pack()

        bounding_label = tk.Label(root, text=f"Bounding Box Type: Straight", padx=10, pady=10)
        bounding_label.pack()

    else:
        print("Not a valid image or video file")
        cap.release()
        cv2.destroyAllWindows()

    # run video after bounding box creation

    # TODO
    """
    TODO: Idea ---> Open a single frame to save the bounding box data, when the window is closed open a another window 
    that will loop the video frames with the bounding block or will watch the video and add the frames to each video
    
    could have prior TODO redundant 
    """

def draw_box(event, x, y, flags, param):
    global bounding_num, frame_copy
    # flags are
    # but param determines the thickness of the boxes
    global ix, iy, drawing # ix, iy is the initial value
    if delete_mode == -1:
        if event == cv2.EVENT_LBUTTONDOWN:
            delete = False
            for i, box in enumerate(bounding_boxes): # this is deletes the object from the bounding box list
                # maybe save the ones created on the actual frame, delete what was created on the frame copy
                x1, y1 = box[3][0][0], box[3][0][1]
                x2, y2 = box[3][1][0], box[3][1][1]

                if x1 < x < x2 and y1 < y < y2:
                    print(box)
                    del bounding_boxes[i]
                    print(bounding_boxes)
                    bounding_num -= 1
                    delete = True


            frame_with_boxes = draw_frame_with_boxes()
            cv2.imshow('First Frame', frame_with_boxes)
    else:
        if event == cv2.EVENT_LBUTTONDOWN: #ix and iy set here
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # frame_copy = frame.copy() # copy is needed since it is temp. drawn on mouse movement
                frame_copy = draw_frame_with_boxes()
                cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 1)
                if value == -1:
                    cv2.circle(frame_copy, ((ix + x) // 2, (iy + y) // 2), 2, (0, 0, 255), -1)  # Add a red dot at the center
                cv2.imshow('First Frame', frame_copy)
                # cv2.imshow('First Frame', )
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            frame_copy = np.zeros_like(frame)  # Create an empty frame
            cv2.imshow('First Frame', frame_copy)
            # cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 1)
        # add a condition that appends a curved box or a straight box similiar to the direction popup
            if value == 1:
                bounding_boxes.append([bounding_num, "straight", direction, [(ix, iy), (x, y)]]) # list(int, string, string, list(2-tuple, (2-tuple)))  #TODO change to choose options
            else:
                bounding_boxes.append([bounding_num, "curved", direction, ((ix, iy), (x, y)), ((ix + x)/2, (iy + y)/2)])
            frame_with_boxes = draw_frame_with_boxes()

        # add text to help with user understanding what box it is
            bounding_num = bounding_num + 1
            print(bounding_boxes[-1])
            print("Straight bounds {b} created at {d} in direction {c}".format(b=bounding_boxes[-1][0], d=(bounding_boxes[-1][3][0], bounding_boxes[-1][3][1]), c=bounding_boxes[-1][2]))
            cv2.imshow('First Frame', frame_with_boxes)

"""
    Function to run the video with the bounding boxes in place  
"""
# TODO include label later
def run_video():
    global cap, bounding_boxes
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1440, 720)) # frame zoomed in without readjustment
        if ret:
            for box in bounding_boxes:
                # box can be
                # either list(boundNum, "straight", direction, list(2-tuple, 2-tuple))
                # or     list(boundNum, "curved", direction, list(2-tuple, 2-tuple) -> boxcords, centercords)
                cv2.rectangle(frame, box[3][0], box[3][1], (0, 255, 0), 1)

            cv2.imshow('Video with Bounding Boxes', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

# gets video from file on computer, mp4 works
select_button = tk.Button(root, text="Select Video File", command=open_file)
select_button.pack()

process_button = tk.Button(root, text="Process video with Bounding Boxes", command=run_video)
process_button.pack()


# tk inter main loop
root.mainloop()
print(bounding_boxes)
