import cv2
import supervision as sv
import os
from ultralytics import YOLO
from roboflow import Roboflow
#If the above imports do not work you will have to pip install each respective directory

def check_file(path : str) -> None:
    '''
    A troubleshooting function to verify if the existing path to a directory exists.
    Input takes a path in a string format and prints out the corresponding attempt.
    Returns None
    '''
    if os.path.isfile(path):
        print("Works")
    else:
        print("Failed")

def load_data():
    '''
    This function uses my API key to load roboflow data from the dropshop dataset.
    Only used once to put the same files in the directory for ease of access.
    '''
    rf = Roboflow(api_key="Izum4d9L9gLJ14xMz3sx")
    project = rf.workspace("dropshop-froos").project("dropshop")
    dataset = project.version(1).download("yolov8")

def main():
    '''
    # model = YOLO('yolov8m.pt')                                           This loads the Training Model
    # model.train(data='DropShop-1\\data.yaml', epochs=5)                  This trains the model and saves into the directory
    # video_path  "droplet_videos\\video_data_Rainbow 11-11-22.m4v"        The next three are just file paths that I checked
    # Dropshop path 'DropShop-1\\data.yaml'
    # Trained data path "runs\detect\\train10\weights\\best.pt"
    
    The above trained data path will be used to load the Yolo model so that you don't have to retrain every time you run.
    This main function is used to test the video detection using SuperVision for annotation
    Box is BoxAnnotator object from SuperVision
    The while loops loads the video and plays it frame by frame while results holds the detected object's data
    My next goal as of 11:29PM 10/29/2023 is to discover is the model can provide a form of movement metric as an attribute
    
    !!!click "ESC" to exit out the function!!!
    '''
    
    model = YOLO("runs\detect\\train10\weights\\best.pt")
    box = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
    video_cap = cv2.VideoCapture("droplet_videos\\video_data_Rainbow 11-11-22.m4v")

    if not video_cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            print("Video is ended")
            break
        
        result = model(frame)[0]
        detections = sv.Detections.from_yolo(result)

        frame = box.annotate(scene=frame, detections=detections)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(0) == 27):
            #!!!click "ESC" to exit out the function!!!
            #Setting waitKey to 0 freezes it at the first frame
            #any number above zero will play the video
            break

if __name__ == '__main__':
    #Important Note run load_data() once
    #Also click "ESC" to exit out the function
    main()
