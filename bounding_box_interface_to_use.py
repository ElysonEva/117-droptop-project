
import cv2
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from pygubu.widgets.editabletreeview import EditableTreeview




class BoundingBoxCreator:
    """
        Class dealing with the interface to create bounding boxes for videos and images passed.

        For use:vid
        stored = BoundingBoxCreator() # menu must be closed, maybe later have button to return the stuff

        stored.return_bounding()
    """
    def __init__(self, cap, master=None):
        self.cap = cap
        self.bounding_boxes = []  # Should be returned to the ML
        self.bounding_num = 0
        self.createBox = True
        self.drawing = False
        self.loaded = False
        self.curve = False
        self.ix, self.iy, self.x, self.y = None, None, None, None
        self.value = 1
        self.firstFrame = 'First Frame'

        self.interface_x_resolution = 1440
        self.interface_y_resolution = 900

        self.display_x_resolution = 640
        self.display_y_resolution = 480

        # directions
        # left (-1,0)
        # right (1, 0)
        # down (0, -1)
        # up (0, 1)
        self.directions = {"Up":(0,0),
                           "Up-Right":(1,1), "Right": (1,0), "Down-Right":(1,-1),
                           "Down":(0,-1),
                           "Down-Left":(-1,-1), "Left":(-1,0), "Up-Left":(-1,1)}
        self.direction = (0, 0)
        self.frame = None
        self.delete_mode = 1
        self.root = tk.Tk() if master is None else tk.Toplevel(master)

        self.root.configure(height=200, width=200)
        self.root.geometry("1024x600")
        # First object created


        frame1 = ttk.Frame(self.root)
        frame1.configure(height=200, width=200)
        self.mainPane = ttk.Panedwindow(frame1, orient="horizontal")
        self.mainPane.configure(height=200, width=200)
        self.mainFrame = ttk.Frame(self.mainPane)
        self.mainFrame.configure(height=200, padding=10, width=200)
        self.boundingFrame = ttk.Frame(self.mainFrame)
        self.boundingFrame.configure(height=200, width=200)
        self.boundingLabel = ttk.Label(self.boundingFrame)
        self.boundingLabel.configure(
            state="normal", text='Bounding Type', width=20)
        self.boundingLabel.grid(
        column=0,
        padx=10,
        pady="15 15",
        row=1,
        sticky="n")

        self.boundingRadioFrame = ttk.Frame(self.boundingFrame)
        self.boundingRadioFrame.configure(height=200, width=200)
        self.straightRadio = ttk.Radiobutton(self.boundingRadioFrame)
        self.straightRadio.configure(text='Straight', value=1)
        self.straightRadio.grid(column=0, row=0, sticky="w")
        self.straightRadio.configure(command=self.set_straight)
        self.curveRadio = ttk.Radiobutton(self.boundingRadioFrame)
        self.curveRadio.configure(text='Curve', value=-1)
        self.curveRadio.grid(column=0, row=1, sticky="w")
        self.curveRadio.configure(command=self.set_curve)
        self.boundingRadioFrame.grid(column=0, row=2)
        self.boundingFrame.grid(column=0, row=0)
        self.attributeFrame = ttk.Frame(self.mainFrame)
        self.attributeFrame.configure(height=200, width=200)
        self.attributeLabel = ttk.Label(self.attributeFrame)
        self.attributeLabel.configure(
            state="normal", text='Attributes\n', width=20)
        self.attributeLabel.grid(column=0, padx=10, pady="15 15", sticky="n")
        self.comboFrame = ttk.Frame(self.attributeFrame)
        self.comboFrame.configure(height=200, width=200)
        self.directionCombo = ttk.Combobox(self.comboFrame, state="readonly", values=list(self.directions.keys()))
        self.Direction = tk.StringVar()
        self.directionCombo.configure(textvariable=self.Direction)
        print(self.direction)
        self.directionCombo.grid(column=0, row=1)
        self.directionCombo.set("Up")
        self.directionCombo.bind("<<ComboboxSelected>>", self.update_direction)

        directionLabel = ttk.Label(self.comboFrame)
        directionLabel.configure(state="normal", text='Direction')
        directionLabel.grid(column=0, padx=5, pady="0 15", row=0, sticky="w")
        self.comboFrame.grid(column=0, padx="10 10", row=3)
        self.attributeFrame.grid(column=0, row=3)
        self.funcFrame = ttk.Frame(self.mainFrame)
        self.funcFrame.configure(height=200, width=200)
        self.clearButton = ttk.Button(self.funcFrame)
        self.clearButton.configure(text='Clear')
        self.clearButton.grid(column=0, padx="10 10", row=1)
        self.clearButton.configure(command=self.clear)
        self.undoButton = ttk.Button(self.funcFrame)
        self.undoButton.configure(text='Undo')
        self.undoButton.grid(column=1, row=1)
        self.undoButton.configure(command=self.undo)
        self.funcFrame.grid(column=0, pady=15, row=5)
        # unused
        # self.buttonDelete = ttk.Frame(self.mainFrame)
        # self.buttonDelete.configure(height=200, width=200)
        # self.deleteButton = ttk.Button(self.buttonDelete)
        # self.deleteButton = ttk.Button(self.buttonDelete, text='Delete Selected Box', command=self.deleteSelected)
        # self.deleteButton.grid(column=1, ipadx=50, pady=100, row=1)
        # self.buttonDelete.grid(column=0, row=6)
        self.mainFrame.grid(column=0, row=0)
        self.mainPane.add(self.mainFrame, weight="1")
        self.editabletreeview1 = EditableTreeview(self.mainPane)
        self.editabletreeview1.configure(selectmode="extended", show="headings")
        editabletreeview1_cols = [
            'tree_bounding_number',
            'tree_bounding_type',
            'tree_bound_direction',
            'tree_bounding_cord']
        editabletreeview1_dcols = [
            'tree_bounding_number',
            'tree_bounding_type',
            'tree_bound_direction',
            'tree_bounding_cord']
        self.editabletreeview1.configure(
            columns=editabletreeview1_cols,
            displaycolumns=editabletreeview1_dcols)
        self.editabletreeview1.column(
            "tree_bounding_number",
            anchor="w",
            stretch=True,
            width=5,
            minwidth=5)
        self.editabletreeview1.column(
            "tree_bounding_type",
            anchor="w",
            stretch=True,
            width=65,
            minwidth=20)
        self.editabletreeview1.column(
            "tree_bound_direction",
            anchor="w",
            stretch=True,
            width=30,
            minwidth=20)
        self.editabletreeview1.column(
            "tree_bounding_cord",
            anchor="w",
            stretch=True,
            width=100,
            minwidth=20)
        self.editabletreeview1.heading(
            "tree_bounding_number",
            anchor="w",
            text='Box #')
        self.editabletreeview1.heading(
            "tree_bounding_type",
            anchor="w",
            text='Bounding Box Type')
        self.editabletreeview1.heading(
            "tree_bound_direction",
            anchor="w",
            text='Direction')
        self.editabletreeview1.heading(
            "tree_bounding_cord",
            anchor="w",
            text='Coordinates')
        self.editabletreeview1.grid(column=0, row=0, sticky="nse")
        self.mainPane.add(self.editabletreeview1, weight="5")
        self.mainPane.grid(column=0, row=0, sticky="nsew")
        frame1.grid(column=0, row=0, sticky="nsew")
        frame1.rowconfigure(0, weight=1)
        frame1.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Main widget
        self.mainwindow = self.root
        # Main menu
        _main_menu = self.create_menu(self.mainwindow)
        self.mainwindow.configure(menu=_main_menu)

        self.root.mainloop()



    def create_menu(self, master):
        menu7 = tk.Menu(master)


        menu7.add("command", command=self.open_file, label='Open Video File')
        menu7.add(
            "command",
            command=self.close_main,
            label='Process Bounding Boxes')
        menu7.add("separator")

        return menu7

    def run(self):
        print("INTERFACE STARTING")
        self.mainwindow.mainloop()
    def clear(self):
        self.bounding_boxes = []
        self.bounding_num = 0
        frame_copy = self.draw_frame_with_boxes()
        cv2.imshow(self.firstFrame, frame_copy)
        self.populate_tree()

    def undo(self):
        if self.bounding_boxes:
            del self.bounding_boxes[-1]
            self.bounding_num -= 1
            frame_copy = self.draw_frame_with_boxes()
            cv2.imshow(self.firstFrame, frame_copy)
            self.populate_tree()

    def populate_tree(self):
        """
        Updates tree with currently added bounding boxes with index number, type, direction, and coordinates.

        """
        self.editabletreeview1.delete(*self.editabletreeview1.get_children())

        for item in self.bounding_boxes:
            # [boundingNum, "straight"|"curved", direction, [(initial x, initial y), (x,y)], OPTIONAL (x, y)]
            self.editabletreeview1.insert("", "end", values=[item[0], item[1], self.directionCombo.get(), item[3]])


    def update_direction(self, event):
        """

        :param event:
        :return:
        """
        selected_direction = self.directionCombo.get()
        self.Direction.set(selected_direction)
        self.direction = self.directions.get(selected_direction, '')




    def set_straight(self):
        self.value = 1

    def set_curve(self):
        self.value = -1


    def toggle_video_loaded(self):
        self.loaded = True if self.loaded is False else True
        # TODO: add button change the text in the statuses


    def deleteSelected(self):
        """"
            Unused, issues with mainloop after deletion

            Also,uneeded since the paths must be in order, use undo for delete

            Deletes selected tree item
        """
        selected = self.editabletreeview1.selection()
        if selected:
            index = self.editabletreeview1.index(selected[-1])
            # self.editabletreeview1.delete(selected)
            self.bounding_boxes.pop(index)
            self.bounding_num -= 1
            self.bounding_boxes = [[idnum] + box[1:] for idnum, box in enumerate(self.bounding_boxes)]
            frame_copy = self.draw_frame_with_boxes()
            cv2.imshow(self.firstFrame, frame_copy)
            self.populate_tree()

        selected = None





    def toggle_delete(self):
        """
        Toggles if the mode is in creation or delete mode. Creation mode (1) allows for creating bounding
        boxes in frame. Deletion mode (-1) allows for deleting the bounding boxes in frame and removal in bounding_box
        """
        self.delete_mode = -self.delete_mode
        typeVal = "Create" if self.delete_mode == 1 else "Delete"

    def open_file(self):
        """
        Ask the user for a video file, opens the first frame to allow for creation of bounding boxes.
        """
        if self.cap.isOpened():
            print("It's a video")
            ret, frame = self.cap.read()
            self.frame = cv2.resize(frame, (self.interface_x_resolution, self.interface_y_resolution))
            cv2.imshow(self.firstFrame, self.frame)
            cv2.setMouseCallback(self.firstFrame, self.draw_box)
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
                    cv2.putText(frame_copy, 'Select Curve End', (10,450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.line(frame_copy, (self.ix, self.iy), (x, y), (0, 255, 0), 1)
                    cv2.imshow(self.firstFrame, frame_copy)
            elif event == cv2.EVENT_LBUTTONUP:
                # creates bounding box based on mouse let go
                self.drawing = False
                self.curve = False
                # frame_copy = np.zeros_like(self.frame)
                frame_with_boxes = self.draw_frame_with_boxes()
                self.bounding_boxes[-1][5] = [(self.ix, self.iy), (x, y)]
                cv2.imshow(self.firstFrame, frame_with_boxes)
                # print("Start:{start}, End: {end}".format(start=(self.ix, self.iy), end=(x,y)))
        else:
            if self.delete_mode == -1:
                if event == cv2.EVENT_LBUTTONDOWN:
                    # handles deletion of the bounding box
                    indices_to_delete = []

                    for i, box in enumerate(self.bounding_boxes):
                        (x1, y1), (x2, y2) = box[3][0], box[3][1]
                        if x1 < x < x2 and y1 < y < y2:
                            print(True)
                            indices_to_delete.append(i)

                    for i in reversed(indices_to_delete):
                        del self.bounding_boxes[i]
                        self.bounding_num -= 1

                    frame_with_boxes = self.draw_frame_with_boxes()
                    cv2.imshow(self.firstFrame, frame_with_boxes)
                    print(self.bounding_boxes)
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
                        cv2.imshow(self.firstFrame, frame_copy)
                elif event == cv2.EVENT_LBUTTONUP:
                    # creates bounding box based on mouse let go
                    self.drawing = False
                    frame_copy = np.zeros_like(self.frame)
                    cv2.imshow(self.firstFrame, frame_copy)

                    # adds bounding box to bounding_boxes based on value type (straight or curved)
                    # forms
                    # [boundingNum, "straight"|"curved", direction, [(initial x, initial y), (x,y)], OPTIONAL (x, y)]

                    # for always getting the TOP LEFT and BOTTOM RIGHT cords
                    top_left = (min(self.ix, x), min(self.iy, y))
                    bottom_right = (max(self.ix, x), max(self.iy, y))
                    if self.value == 1:
                        self.bounding_boxes.append([self.bounding_num, "straight", self.direction, [top_left, bottom_right]])
                        print(self.bounding_boxes[-1])
                    else:
                        self.curve = True
                        midpoint = ((self.ix + x) / 2, (self.iy + y) / 2)
                        self.bounding_boxes.append([self.bounding_num, "curved", self.direction, [top_left, bottom_right], midpoint, [(0, 0), (0, 0)]])
                        print(self.bounding_boxes[-1])
                    self.populate_tree()
                    frame_with_boxes = self.draw_frame_with_boxes()
                    self.bounding_num = self.bounding_num + 1
                    # print(
                    #     f"Straight bounds {self.bounding_boxes[-1][0]} created at {self.bounding_boxes[-1][3][0]}, {self.bounding_boxes[-1][3][1]} in direction {self.bounding_boxes[-1][2]}")
                    cv2.imshow(self.firstFrame, frame_with_boxes)

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

    # def run_video(self): # UNUSED
    #     """
    #     Runs the video with the bounding boxes included, Press 'q' to exit,
    #
    #     """
    #     while self.cap.isOpened():
    #         ret, frame = self.cap.read()
    #         # original_width = int(cap.get(3))  # Width
    #         # original_height = int(cap.get(4))
    #         frame = cv2.resize(frame, (1200, 720))
    #         if ret:
    #             for box in self.bounding_boxes:
    #                 if box[1] == "straight":
    #                     cv2.rectangle(frame, box[3][0], box[3][1], (0, 255, 0), 1)
    #                 elif box[1] == "curved":
    #                     cv2.rectangle(frame, box[3][0], box[3][1], (0, 255, 0), 1)
    #                     cv2.circle(frame, (int(box[4][0]), int(box[4][1])), 2, (0, 0, 255), -1)
    #
    #             cv2.imshow('Video with Bounding Boxes', frame)
    #
    #             if cv2.waitKey(25) & 0xFF == ord('q'):
    #                 break
    #         else:
    #             break
    #     self.cap.release()
    #     cv2.destroyAllWindows()

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
        used to scale the bounding boxes in the droplet tracker
        """

        self.display_x_resolution = x # 640
        self.display_y_resolution = y # 480

        x_scale = self.display_x_resolution / self.interface_x_resolution # 640/1440 = .4444
        y_scale = self.display_y_resolution / self.interface_y_resolution # 480/900 = .5333

        for i, box in enumerate(self.bounding_boxes):
            # TODO: I am resizing the frame based on the scale of the video interface and the video show,
            # TODO: need to
            # cv2 create rectangle needs int values for cords
            box[3][0] = tuple([int(box[3][0][0] * x_scale), int(box[3][0][1] * y_scale)])
            box[3][1] = tuple([int(box[3][1][0] * x_scale), int(box[3][1][1] * y_scale)])

            if box[1] == "Curve":
                box[4] = ((box[3][0][0] + box[3][1][0]) / 2, (box[3][0][1] + box[3][1][1]))

    def close_main(self):
        self.root.destroy()







