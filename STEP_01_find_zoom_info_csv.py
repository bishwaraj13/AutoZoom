import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
import os

# Create directories if they don't exist
output_dir = Path('heatmap_outputs_video')
output_dir.mkdir(exist_ok=True)


def compute_optical_flow(prev_frame, current_frame, margin_percent=10):
    """Compute optical flow between two frames with margin masking."""
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate frame dimensions
    frame_height, frame_width = prev_gray.shape
    margin_x = int(frame_width * margin_percent / 100)
    margin_y = int(frame_height * margin_percent / 100)

    # Create margin mask
    margin_mask = np.ones((frame_height, frame_width), dtype=bool)
    margin_mask[:margin_y, :] = False
    margin_mask[-margin_y:, :] = False
    margin_mask[:, :margin_x] = False
    margin_mask[:, -margin_x:] = False

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, current_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = magnitude * margin_mask

    return magnitude, angle, flow


def create_heatmap(magnitude, sigma=5):
    """Create a smoothed heatmap from flow magnitude."""
    smoothed = gaussian_filter(magnitude, sigma=sigma)
    normalized = (smoothed - smoothed.min()) / \
        (smoothed.max() - smoothed.min() + 1e-8)
    return normalized


def find_activity_threshold(heatmap, sensitivity=0.5):
    """Calculate activity threshold for the heatmap."""
    mean_activity = np.mean(heatmap)
    std_activity = np.std(heatmap)
    return mean_activity + sensitivity * std_activity


def create_binary_mask(heatmap, threshold):
    """Create binary mask from heatmap using threshold."""
    return (heatmap > threshold).astype(np.uint8) * 255


def find_bounding_rectangles(binary_mask, min_area=100, frame_number=None, total_area=None):
    """Find bounding rectangles around active regions, excluding those contained within larger rectangles."""
    orig_width, orig_height = binary_mask.shape
    contours, _ = cv2.findContours(binary_mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # First, get all valid rectangles
    all_rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out small rectangles
            if w * h * 100 / total_area > 0.9:  # Ignore
                all_rectangles.append((x, y, w, h))

    # Filter out contained rectangles
    filtered_rectangles = []
    for i, rect1 in enumerate(all_rectangles):
        x1, y1, w1, h1 = rect1
        is_contained = False

        # Check if this rectangle is contained within any other rectangle
        for j, rect2 in enumerate(all_rectangles):
            if i != j:  # Don't compare with itself
                x2, y2, w2, h2 = rect2

                # Check if rect1 is contained within rect2
                if (x1 >= x2 and y1 >= y2 and
                    x1 + w1 <= x2 + w2 and
                        y1 + h1 <= y2 + h2):
                    is_contained = True
                    break

        if not is_contained:
            filtered_rectangles.append(rect1)

    # Find the largest rectangle among filtered rectangles
    largest_rect = None
    largest_area = 0
    df_row = None

    for x, y, w, h in filtered_rectangles:
        rect_area = w * h
        if rect_area > largest_area:
            largest_area = rect_area
            largest_rect = (x, y, w, h)

    # Create DataFrame row if we have a large enough rectangle
    if largest_rect and total_area and frame_number is not None:
        area_percentage = (largest_area / total_area) * 100

        if area_percentage > 1:  # Check if larger than 1% of total area
            x, y, w, h = largest_rect

            df_row = {
                'frame': frame_number,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'focus': True,
                'area_percentage': area_percentage,
                # Count of non-contained rectangles
                'activity_roi_count': len(filtered_rectangles)
            }

    return filtered_rectangles, df_row


def draw_rectangles(image, rectangles, color=(0, 255, 0), thickness=2):
    """Draw rectangles on an image."""
    result = image.copy()
    for x, y, w, h in rectangles:
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def get_final_heatmap_visuals(prev_frame, current_frame, frame_number):
    """Get final heatmap visuals for two consecutive frames."""
    # Compute optical flow
    magnitude, angle, flow = compute_optical_flow(prev_frame, current_frame)

    # Create heatmap
    heatmap = create_heatmap(magnitude)

    # Get total frame area
    total_area = prev_frame.shape[0] * prev_frame.shape[1]

    # Find activity threshold and create binary mask
    threshold = find_activity_threshold(heatmap, sensitivity=0.5)
    binary_mask = create_binary_mask(heatmap, threshold)

    # Get rectangles and potential DataFrame row
    rectangles, df_row = find_bounding_rectangles(binary_mask, min_area=100,
                                                  frame_number=frame_number,
                                                  total_area=total_area)

    # Create visualization
    normalized_heatmap = (heatmap - heatmap.min()) / \
        (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_colored = plt.cm.jet(normalized_heatmap)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
    result = draw_rectangles(heatmap_bgr, rectangles)

    # Save the plot
    if df_row is not None:  # Only save if we have a significant rectangle
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(output_dir / f'heatmap_frame_{frame_number}.png')
        plt.close()

    return result, df_row


def process_video(video_path):
    """Process video file and extract frames every 0.5 seconds."""
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Number of frames to skip for 0.5 second intervals
    frame_interval = int(fps * 0.5)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(
        columns=['frame', 'x', 'y', 'width', 'height', 'focus', 'area_percentage', 'activity_roi_count'])

    # Initialize variables for frame processing
    frame_count = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if prev_frame is not None:
                # Process frame pair
                current_frame_number = frame_count // frame_interval
                result, df_row = get_final_heatmap_visuals(
                    prev_frame, frame, current_frame_number)

                if df_row is not None:
                    results_df.loc[len(results_df)] = df_row

            prev_frame = frame.copy()

        frame_count += 1

        # Optional: Display progress
        if frame_count % fps == 0:
            print(f"Processing... {frame_count /
                  total_frames*100: .1f} % complete")

    # Release resources
    cap.release()

    # Save results to CSV
    results_df.to_csv('heatmap_results_old.csv', index=False)
    print("Processing complete. Results saved to 'heatmap_results.csv'")


# Example usage:
process_video(
    "/Users/bishwaraj/Documents/screencast-refiner/composite_videos/fdebd853-a815-46ad-bbd2-335471d2d640_composite.mp4")
