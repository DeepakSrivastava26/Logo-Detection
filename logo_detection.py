import numpy as np
import torch
from torchvision import transforms, models
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Load the VGG16 model for feature extraction
vgg = models.vgg16(pretrained=True).features.eval()

def preprocess_image(image, size=(224, 224)):
    """Preprocess image for VGG16."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def extract_features(image_tensor, model, layer_index=29):
    """Extract features from a specific VGG16 layer."""
    with torch.no_grad():
        features = model[:layer_index](image_tensor)
    return features.squeeze(0)

def extract_patches(target_image, patch_size, stride=16):
    """Extract patches from the target image."""
    patches = []
    h, w, _ = target_image.shape
    patch_w, patch_h = patch_size
    for y in range(0, h - patch_h + 1, stride):
        for x in range(0, w - patch_w + 1, stride):
            patch = target_image[y:y + patch_h, x:x + patch_w]
            patches.append((patch, (x, y)))
    return patches

def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression to filter overlapping boxes."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def template_matching(target, template, stride=16, threshold=0.9, iou_threshold=0.1):
    """
    Match template in the target image using feature similarity.
    Returns bounding boxes of matched regions.
    """
    h_template, w_template, _ = template.shape
    template_tensor = preprocess_image(template, size=(h_template, w_template))
    template_features = extract_features(template_tensor, vgg)

    patches = extract_patches(target, (w_template, h_template), stride)
    boxes, scores = [], []

    for patch, loc in patches:
        patch_tensor = preprocess_image(patch, size=(h_template, w_template))
        patch_features = extract_features(patch_tensor, vgg)
        sim = cosine_similarity(template_features.flatten().reshape(1, -1),
                                patch_features.flatten().reshape(1, -1))[0][0]
        if sim >= threshold:
            x1, y1 = loc
            x2, y2 = x1 + w_template, y1 + h_template
            boxes.append((x1, y1, x2, y2))
            scores.append(sim)

    keep_indices = nms(boxes, scores, iou_threshold)
    final_matches = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]) for i in keep_indices]
    return final_matches

def draw_boxes(image, matches):
    """Draw bounding boxes on the image."""
    for (x1, y1, x2, y2, score) in matches:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    return image

def detect_objects(video_path, template_path):
    """
    Detect objects in the first frame of the video using template matching.
    Returns bounding box information.
    """
    # Load template image
    template_image = cv2.imread(template_path)
    if template_image is None:
        raise FileNotFoundError("Template image not found at the specified path.")

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame of the video.")
    cap.release()

    # Perform template matching
    matches = template_matching(frame, template_image, stride=32, threshold=0.93, iou_threshold=0.1)

    # Draw boxes on the first frame for visualization (optional)
    output_frame = draw_boxes(frame, matches)
    cv2.imwrite("output_frame_with_boxes.jpg", output_frame)

    # Return bounding box information
    bounding_boxes = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2, _) in matches]  # Convert to (x, y, w, h)
    return bounding_boxes
