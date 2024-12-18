import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import os

# VGG Feature Extractor
class VGGFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.vgg16(pretrained=True).features.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return features.cpu().numpy()

# Non-Maximum Suppression (NMS)
def non_maximum_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    pick = []
    while len(indices) > 0:
        last = indices[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[indices[:-1]])
        yy1 = np.maximum(y1[last], y1[indices[:-1]])
        xx2 = np.minimum(x2[last], x2[indices[:-1]])
        yy2 = np.minimum(y2[last], y2[indices[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[indices[:-1]]

        indices = indices[np.where(overlap <= overlap_thresh)[0]]

    return boxes[pick].astype("int")

# Draw bounding boxes interactively
def draw_bounding_boxes(frame):
    bounding_boxes = []
    window_name = "Draw Bounding Boxes (Press 'q' to quit, 's' to save)"
    clone = frame.copy()

    def draw_rectangle(event, x, y, flags, param):
        nonlocal start_x, start_y, drawing, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            start_x, start_y = x, y
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            clone = frame.copy()
            cv2.rectangle(clone, (start_x, start_y), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(clone, (start_x, start_y), (x, y), (0, 255, 0), 2)
            bounding_boxes.append((start_x, start_y, x, y))

    start_x = start_y = 0
    drawing = False
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)

    while True:
        cv2.imshow(window_name, clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        elif key == ord("q"):
            bounding_boxes = []
            break

    cv2.destroyWindow(window_name)
    return bounding_boxes

# Compute homography and inverse transform
def compute_inverse_homography(ref_frame, curr_frame):
    gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if len(kp1) <4 or len(kp2) <4:
        print("Not enough keypoints detected")
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print("Not enough matches to compute homography.")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if H is not None:
        height, width = ref_frame.shape[:2]
        warped_frame = cv2.warpPerspective(curr_frame, H, (width, height))
        return H, warped_frame
    else:
        print("Homography computation failed.")
        return None, None

# VGG-based Template Matching
def vgg_template_matching(vgg_extractor, ref_bbox, warped_image):
    x1, y1, x2, y2 = ref_bbox
    patch = warped_image[y1:y2, x1:x2]

    if patch.size == 0:
        return []

    ref_features = vgg_extractor.extract_features(patch)
    height, width = warped_image.shape[:2]
    step_size =  64 # Sliding window step size
    matched_boxes = []

    for y in range(0, height - (y2 - y1), step_size):
        for x in range(0, width - (x2 - x1), step_size):
            window = warped_image[y:y + (y2 - y1), x:x + (x2 - x1)]
            if window.size == 0:
                continue

            window_features = vgg_extractor.extract_features(window)
            distance = np.linalg.norm(ref_features - window_features)
            if distance < 5.0:  # Threshold for matching
                matched_boxes.append((x, y, x + (x2 - x1), y + (y2 - y1)))

    return matched_boxes

# Main Program
def main(video_path, output_folder,frames_to_process):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    ret, ref_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return

    print("Draw bounding boxes on the reference frame...")
    bounding_boxes = draw_bounding_boxes(ref_frame)
    vgg_extractor = VGGFeatureExtractor()

    frame_idx = 0
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        if frame_idx in frames_to_process:
            print(f"Processing frame {frame_idx}...")
            _, ref_hat = compute_inverse_homography(ref_frame, curr_frame)
            if ref_hat is not None:
                all_boxes = []
                for box in bounding_boxes:
                    matched_boxes = vgg_template_matching(vgg_extractor, box, ref_hat)
                    all_boxes.extend(matched_boxes)

                final_boxes = non_maximum_suppression(all_boxes)
                for (x1, y1, x2, y2) in final_boxes:
                    print(f"Final boxes values are",final_boxes)
                    cv2.rectangle(ref_hat, (x1, y1), (x2, y2), (0, 0, 255), 2)

                output_path = f"{output_folder}/bbox_frame_{frame_idx:04d}.jpg"
                cv2.imwrite(output_path, ref_hat)

        frame_idx += 1

    cap.release()
    print("Processing completed.")

if __name__ == "__main__":
    video_path = "x.mp4"
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)
    frames_to_process = [16]
    main(video_path, output_folder,frames_to_process)
