# import packages
import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

file_paths = []
output_folders = []

def extract_frames(video_path, output_folder, frame_rate):
    """
    Function to extract frames from a given video and save them as .png in the respective directory.
    """

    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame{frame_count}.png"), frame)
            frame_count += 1

        count += 1

    cap.release()
    return frame_count

def select_videos():
    """
    Function to select video files and create corresponding output folders.
    """
    global output_folders
    global file_paths
    file_paths = (filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4"), ("Video files", "*.mov")]))
    video_paths_var.set(file_paths)

    # Create output folders based on video file names

    for file_path in file_paths:
        folder_name = os.path.splitext(os.path.basename(file_path))[0]

        output_folder = os.path.join("output", folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_folders.append(output_folder)

def start_extraction():
    """
    Function to start the frame extraction process.
    """
    try:
        frame_rate = int(frame_rate_var.get())
        print(output_folders)

        for video_path, output_folder in zip(file_paths, output_folders):
            print(video_path)
            print(output_folder)
            num_frames = extract_frames(video_path, output_folder, frame_rate)
            status_text_var.set(f"Extracted {num_frames} frames from {os.path.basename(video_path)}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid frame rate.")

if __name__ == "__main__":

    root = tk.Tk()
    root.title("Frame Extractor")

    # Variables
    video_paths_var = tk.StringVar()
    output_folders_var = tk.StringVar()
    frame_rate_var = tk.StringVar()
    status_text_var = tk.StringVar()

    # Layout
    tk.Button(root, text="Select Videos", command=select_videos).pack()
    tk.Entry(root, textvariable=video_paths_var, width=50).pack()

    tk.Label(root, text="Frame Rate:").pack()
    tk.Entry(root, textvariable=frame_rate_var).pack()

    tk.Button(root, text="Start Extraction", command=start_extraction).pack()
    tk.Label(root, textvariable=status_text_var).pack()

    # Run the application
    root.mainloop()