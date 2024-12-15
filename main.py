'''from extract_frames import extract_frames
from logo_detection import detect_objects
from csrt import track_with_csrt
import os
import cv2

def main(video_path, output_frames_folder):
    # Step 1: Extract frames
    print("Extracting frames...")
    extract_frames(video_path, output_frames_folder)
    
    # Step 2: Object detection on the first frame
    print("Running object detection...")
    first_frame_path = os.path.join(output_frames_folder, "frame_0000.jpg")
    first_frame = cv2.imread(first_frame_path)
    detected_objects = detect_objects(first_frame)

    if detected_objects:
        # Convert detected objects to bounding box format (x, y, w, h)
        bounding_boxes = [
            (obj["x"], obj["y"], obj["w"], obj["h"]) for obj in detected_objects
        ]
        print(f"Detected {len(bounding_boxes)} object(s): {bounding_boxes}")
        
        # Step 3: Start CSRT tracking
        print("Starting CSRT tracker for multiple objects...")
        track_with_csrt(video_path, bounding_boxes, start_frame=0)
    else:
        print("No objects detected in the first frame.")

if __name__ == "__main__":
    video_path = "input_video.mp4"
    output_frames_folder = "extracted_frames"
    main(video_path, output_frames_folder)'''

from frame_extraction import extract_frames
from logo_detection import detect_objects  # Updated detection function
from csrt import track_with_csrt
import os
import cv2

def main(video_path, template_path, output_frames_folder):
    # Step 1: Extract frames
    print("Extracting frames...")
    extract_frames(video_path, output_frames_folder)
    
    # Step 2: Object detection on the first frame
    print("Running object detection on the first frame...")
    first_frame_path = os.path.join(output_frames_folder, "frame_0000.jpg")
    first_frame = cv2.imread(first_frame_path)

    if first_frame is None:
        print("Error: First frame not found or unreadable.")
        return

    # Detect objects using template matching
    detected_bounding_boxes = detect_objects(video_path, template_path)

    if detected_bounding_boxes:
        print(f"Detected {len(detected_bounding_boxes)} object(s): {detected_bounding_boxes}")
        
        # Step 3: Start CSRT tracking for the detected bounding boxes
        print("Starting CSRT tracker for multiple objects...")
        track_with_csrt(video_path, detected_bounding_boxes, output_video_path, start_frame=0)
    else:
        print("No objects detected in the first frame.")

if __name__ == "__main__":
    video_path = "x.mp4"       # Path to the input video file
    template_path = "t1.png"              # Path to the template image
    output_frames_folder = "extracted_frames"  # Folder to save extracted frames
    output_video_path = "output.avi"
    
    main(video_path, template_path, output_frames_folder)

