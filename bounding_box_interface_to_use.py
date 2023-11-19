import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np


class BoundingBoxCreator:
    """
        Class dealing with the interface to create bounding boxes for videos and images passed.

        For use:vid
        stored = BoundingBoxCreator() # menu must be closed, maybe later have button to return the stuff

        stored.return_bounding()
    """
    def __init__(self, cap):
        self.cap = cap
        self.bounding_boxes = []  # Should be returned to the ML
        self.bounding_num = 1
        self.createBox = True
        self.drawing = False
        self.loaded = False
        self.curve = False
        self.ix, self.iy, self.x, self.y = None, None, None, None
        self.value = 1

        self.interface_x_resolution = 1440
        self.interface_y_resolution = 900

        self.display_x_resolution = 640
        self.display_y_resolution = 480

        # TODO change how direction is implemented to (0, 0) system
        self.direction = "Up"
        self.frame = None
        self.delete_mode = 1

        # creation of the root for the bound box interface and the starts the creation of interface
        # TODO make not BAD
        self.root = tk.Tk()
        self.root.geometry("400x200")
        self.root.title('Bounding box creation: Please Select bounding boxes')

        self.videoFrame = tk.Frame(border=1, padx=10)

        self.select_button = tk.Button(self.videoFrame, text="Select Video File", command=self.open_file)
        self.select_button.pack()

        # self.process_button = tk.Button(self.videoFrame, text="Process video with Bounding Boxes",
        #                                 command=self.run_video)

        self.bounding = tk.Button(self.videoFrame, text="Return the bounding boxes",
                                        command=self.close_main)

        self.bounding.pack()

        self.videoFrame.pack()

        # self.bounding.pack()

        self.typeButton = tk.Button(self.root, text="Bounding Box Type: Straight", command=self.toggle_value, pady=20)
        self.typeButton.pack()

        self.modeButton = tk.Button(self.root, text="Mode: Create", command=self.toggle_delete, pady=20)
        self.modeButton.pack()

        self.direction_menu = tk.Menu(self.root)
        self.root.config(menu=self.direction_menu)

        self.direction_submenu = tk.Menu(self.direction_menu, tearoff=0)
        # TODO: change how direction to have up and down
        self.direction_menu.add_cascade(label="Direction", menu=self.direction_submenu)
        self.direction_menu.add_command(label="Up", command=lambda: self.set_direction("Up"))
        self.direction_menu.add_command(label="Down", command=lambda: self.set_direction("Down"))
        self.direction_menu.add_command(label="Left", command=lambda: self.set_direction("Left"))
        self.direction_menu.add_command(label="Right", command=lambda: self.set_direction("Right"))

        self.option_frame = tk.Frame(self.root)

        self.direction_label = tk.Label(self.option_frame, text=f"Selected Direction: {self.direction}", padx=10,
                                        pady=10)
        self.direction_label.pack()

        self.option_frame.pack()

        self.root.mainloop()

    def set_direction(self, value):
        """
        Set the direction of the direction for the bounding box.

        :param value: Direction of the bounding box, currently takes values "Up", "Down", "Left", and "Right"'
        """

        self.direction = value
        self.direction_label.config(text=f"Selected Direction: {self.direction}")

    def toggle_value(self):
        """
        Toggles if the bounding box is a 'Straight' or 'Curve.' Makes changes to the button text reflecting
        the current type.
        """
        self.value = -self.value
        typeVal = "Straight" if self.value == 1 else "Curve"
        self.typeButton.config(text=f"Bounding Box Type: {typeVal}")

    def toggle_video_loaded(self):
        self.loaded = True if self.loaded is False else True
        # TODO: add button change the text in the statuses

    def toggle_delete(self):
        """
        Toggles if the mode is in creation or delete mode. Creation mode (1) allows for creating bounding
        boxes in frame. Deletion mode (-1) allows for deleting the bounding boxes in frame and removal in bounding_box
        """
        self.delete_mode = -self.delete_mode
        typeVal = "Create" if self.delete_mode == 1 else "Delete"
        self.modeButton.config(text=f"Mode: {typeVal}")

    def open_file(self):
        """
        Ask the user for a video file, opens the first frame to allow for creation of bounding boxes.
        """
        if self.cap.isOpened():
            print("It's a video")
            ret, frame = self.cap.read()
            self.frame = cv2.resize(frame, (self.interface_x_resolution, self.interface_y_resolution))
            cv2.imshow('First Frame', self.frame)
            cv2.setMouseCallback('First Frame', self.draw_box)
        else:
            print("Not a valid image or video file")
            self.cap.release()
            cv2.destroyAllWindows()

    def draw_box(self, event, x, y, flags, param):
        """
        Handles interactions with both box creation and deletion. May need separation later but both events
        are combined.

        :param event: what event is pushed (mostly for mouse interactions)
        :param x: x position for mouse
        :param y: y position for mouse
        :param flags: unused but returned from event press
        :param param: unused but returned from event press
        """
        if self.curve is True:
            # used for getting the start and end of the curved box
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                # creates reference bounding box on mouse movement
                if self.drawing:
                    frame_copy = self.draw_frame_with_boxes()
                    cv2.line(frame_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 1)
                    cv2.imshow('First Frame', frame_copy)
            elif event == cv2.EVENT_LBUTTONUP:
                # creates bounding box based on mouse let go
                self.drawing = False
                self.curve = False
                frame_copy = np.zeros_like(self.frame)
                frame_with_boxes = self.draw_frame_with_boxes()
                self.bounding_boxes[-1][5] = [(self.ix, self.iy), (x, y)]
                cv2.imshow('First Frame', frame_with_boxes)
        else:
            if self.delete_mode == -1:
                if event == cv2.EVENT_LBUTTONDOWN:
                    # handles deletion of the bounding box
                    for i, box in enumerate(self.bounding_boxes):
                        x1, y1 = box[3][0][0], box[3][0][1]
                        x2, y2 = box[3][1][0], box[3][1][1]

                        if x1 < x < x2 and y1 < y < y2:
                            del self.bounding_boxes[i]
                            self.bounding_num -= 1
                    frame_with_boxes = self.draw_frame_with_boxes()
                    cv2.imshow('First Frame', frame_with_boxes)
            else:
                if event == cv2.EVENT_LBUTTONDOWN:
                    # handles obtaining the initial coordinate position for bounding box
                    self.drawing = True
                    self.ix, self.iy = x, y
                elif event == cv2.EVENT_MOUSEMOVE:
                    # creates reference bounding box on mouse movement
                    if self.drawing:
                        frame_copy = self.draw_frame_with_boxes()
                        cv2.rectangle(frame_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 1)
                        # center dot
                        if self.value == -1:
                            cv2.circle(frame_copy, ((self.ix + x) // 2, (self.iy + y) // 2), 2, (0, 0, 255), -1)
                        cv2.imshow('First Frame', frame_copy)
                elif event == cv2.EVENT_LBUTTONUP:
                    # creates bounding box based on mouse let go
                    self.drawing = False
                    frame_copy = np.zeros_like(self.frame)
                    cv2.imshow('First Frame', frame_copy)

                    # adds bounding box to bounding_boxes based on value type (straight or curved)
                    # forms
                    # [boundingNum, "straight"|"curved", direction, [(initial x, initial y), (x,y)], OPTIONAL (x, y)]
                    if self.value == 1:
                        self.bounding_boxes.append(
                            [self.bounding_num, "straight", self.direction, [(self.ix, self.iy), (x, y)]])
                    else:
                        self.curve = True
                        self.bounding_boxes.append(
                            [self.bounding_num, "curved", self.direction, [(self.ix, self.iy), (x, y)],
                             [(self.ix + x) / 2, (self.iy + y) / 2], [(0,0),(0,0)]])



                    frame_with_boxes = self.draw_frame_with_boxes()
                    self.bounding_num = self.bounding_num + 1
                    print(
                        f"Straight bounds {self.bounding_boxes[-1][0]} created at {self.bounding_boxes[-1][3][0]}, {self.bounding_boxes[-1][3][1]} in direction {self.bounding_boxes[-1][2]}")
                    cv2.imshow('First Frame', frame_with_boxes)

    def draw_frame_with_boxes(self):
        """
        Draws the frames that are in bounding_boxes

        :return: frame with all the bounding boxes included
        """
        temp_frame = self.frame.copy()
        for box in self.bounding_boxes:
            box[0] = self.bounding_boxes.index(box)
            x1, y1 = box[3][0][0], box[3][0][1]
            x2, y2 = box[3][1][0], box[3][1][1]
            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # puts text of what order the bounding box is
            cv2.putText(temp_frame, str(box[0]), ((x1 + x2) // 2, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)  # Add number label

        return temp_frame

    def run_video(self):
        """
        Runs the video with the bounding boxes included, Press 'q' to exit
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            # original_width = int(cap.get(3))  # Width
            # original_height = int(cap.get(4))
            frame = cv2.resize(frame, (1200, 720))
            if ret:
                for box in self.bounding_boxes:
                    if box[1] == "straight":
                        cv2.rectangle(frame, box[3][0], box[3][1], (0, 255, 0), 1)
                    elif box[1] == "curved":
                        cv2.rectangle(frame, box[3][0], box[3][1], (0, 255, 0), 1)
                        cv2.circle(frame, (int(box[4][0]), int(box[4][1])), 2, (0, 0, 255), -1)

                cv2.imshow('Video with Bounding Boxes', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()


    def close_main(self):
        self.root.destroy()
    def return_bounding(self):
        """
        Returns the bounding boxes in bounding_boxes
        :return: all bounding boxes in self.bounding_boxes
        """

        self.cap.release()
        cv2.destroyAllWindows()
        return self.bounding_boxes

    def set_resolution(self, x, y):
        """"
        Returns the resolution of the bounding box interface creator,
        used to scale the bounding boxes in the droplet tracker
        """

        self.display_x_resolution = x
        self.display_y_resolution = y

        x_scale = self.display_x_resolution / self.interface_x_resolution
        y_scale = self.display_y_resolution / self.interface_y_resolution

        for i, box in enumerate(self.bounding_boxes):
            print(box[3])
            box[3][0] = tuple([int(box[3][0][0] * x_scale), int(box[3][0][1] * y_scale)])
            box[3][1] = tuple([int(box[3][1][0] * x_scale), int(box[3][1][1] * y_scale)])
            print(box[3])

            if box[1] == "Curve":
                box[4] = ((box[3][0][0] + box[3][1][0]) / 2, (box[3][0][1] + box[3][1][1]))



