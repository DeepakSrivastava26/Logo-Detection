'''import cv2

def track_with_csrt(video_path, bounding_boxes, start_frame=0):
    """
    Tracks multiple objects using CSRT trackers.

    Parameters:
    - video_path (str): Path to the input video.
    - bounding_boxes (list): List of bounding boxes as [(x, y, w, h), ...].
    - start_frame (int): Frame number to start tracking from.
    """
    cap = cv2.VideoCapture(video_path)

    # Initialize multiple trackers
    trackers = []
    for bbox in bounding_boxes:
        tracker = cv2.TrackerCSRT_create()
        trackers.append({"tracker": tracker, "bbox": bbox})

    # Move to the specified start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read the starting frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video at the start frame.")
        return

    # Initialize trackers on the starting frame
    for t in trackers:
        t["tracker"].init(frame, t["bbox"])

    # Start tracking
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update each tracker and draw bounding boxes
        for t in trackers:
            success, bbox = t["tracker"].update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                t["bbox"] = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("Tracking failed for one of the objects.")
        
        # Display the frame with all tracked objects
        cv2.imshow("Multiple CSRT Tracking", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()'''
import cv2

def track_with_csrt(video_path, bounding_boxes, output_video_path, start_frame=0):
    """
    Tracks multiple objects using CSRT trackers and saves the output to a video file.

    Parameters:
    - video_path (str): Path to the input video.
    - bounding_boxes (list): List of bounding boxes as [(x, y, w, h), ...].
    - output_video_path (str): Path to save the output video.
    - start_frame (int): Frame number to start tracking from.
    """
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    # Get video properties (frame width, height, FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (e.g., XVID, MJPG, MP4V)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize multiple CSRT trackers
    trackers = []
    for bbox in bounding_boxes:
        tracker = cv2.TrackerCSRT_create()
        trackers.append({"tracker": tracker, "bbox": bbox})

    # Move to the specified start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read the starting frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read the video at the start frame.")
        cap.release()
        out.release()
        return

    # Initialize trackers on the starting frame
    for t in trackers:
        t["tracker"].init(frame, t["bbox"])

    # Start tracking and writing to output
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update each tracker and draw bounding boxes
        for t in trackers:
            success, bbox = t["tracker"].update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                t["bbox"] = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("Warning: Tracking failed for one of the objects.")

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")

