import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoHeatmapAnalyzer:
    """
    A class for analyzing video content and generating activity heatmaps.

    Attributes:
        output_dir (Path): Directory for storing output files
        min_area (int): Minimum area threshold for activity detection
        sensitivity (float): Sensitivity threshold for activity detection
        heatmap_subdir (Path): Subdirectory for storing heatmap images
    """

    def __init__(
        self,
        output_dir: str = 'heatmap_outputs',
        min_area: int = 100,
        sensitivity: float = 0.5
    ):
        """
        Initialize the VideoHeatmapAnalyzer.

        Args:
            output_dir (str): Directory path for output files
            min_area (int): Minimum area threshold for activity detection
            sensitivity (float): Sensitivity threshold for activity detection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.heatmap_subdir = self.output_dir / 'heatmap_frames'
        self.min_area = min_area
        self.sensitivity = sensitivity

    def compute_optical_flow(
        self,
        prev_frame: np.ndarray,
        current_frame: np.ndarray,
        margin_percent: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute optical flow between two consecutive frames with margin masking.

        Args:
            prev_frame: Previous video frame
            current_frame: Current video frame
            margin_percent: Percentage of frame to mask at edges

        Returns:
            Tuple containing:
            - magnitude: Magnitude of optical flow
            - angle: Angle of optical flow
            - flow: Raw optical flow data
        """
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame dimensions and margins
        frame_height, frame_width = prev_gray.shape
        margin_x = int(frame_width * margin_percent / 100)
        margin_y = int(frame_height * margin_percent / 100)

        # Create margin mask
        margin_mask = np.ones((frame_height, frame_width), dtype=bool)
        margin_mask[:margin_y, :] = False
        margin_mask[-margin_y:, :] = False
        margin_mask[:, :margin_x] = False
        margin_mask[:, -margin_x:] = False

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            current_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Convert flow to polar coordinates
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitude = magnitude * margin_mask

        return magnitude, angle, flow

    def create_heatmap(
        self,
        magnitude: np.ndarray,
        sigma: float = 5
    ) -> np.ndarray:
        """
        Create a smoothed heatmap from flow magnitude.

        Args:
            magnitude: Optical flow magnitude array
            sigma: Gaussian smoothing parameter

        Returns:
            Normalized heatmap array
        """
        smoothed = gaussian_filter(magnitude, sigma=sigma)
        normalized = (smoothed - smoothed.min()) / \
            (smoothed.max() - smoothed.min() + 1e-8)
        return normalized

    def draw_rectangles(
        self,
        image: np.ndarray,
        rectangles: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw rectangles on an image.

        Args:
            image: Input image
            rectangles: List of rectangle coordinates (x, y, width, height)
            color: BGR color tuple for rectangles
            thickness: Line thickness for rectangles

        Returns:
            Image with rectangles drawn
        """
        result = image.copy()
        for x, y, w, h in rectangles:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        return result

    def save_heatmap_visualization(
        self,
        heatmap: np.ndarray,
        rectangles: List[Tuple[int, int, int, int]],
        frame_number: int,
        timestamp: float
    ) -> None:
        """
        Save visualization of heatmap with rectangles.

        Args:
            heatmap: Normalized heatmap array
            rectangles: List of rectangle coordinates
            frame_number: Current frame number
            timestamp: Current video timestamp in seconds
        """
        # Convert heatmap to color visualization
        heatmap_colored = plt.cm.jet(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

        # Draw rectangles on heatmap
        result = self.draw_rectangles(heatmap_bgr, rectangles)

        # Save visualization with timestamp in filename
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'Time: {timestamp:.2f}s')
        plt.axis('off')
        plt.savefig(self.heatmap_subdir /
                    f'heatmap_frame_{frame_number}_{timestamp:.2f}s.png')
        plt.close()

    def find_activity_regions(
        self,
        heatmap: np.ndarray,
        frame_number: int,
        timestamp: float,
        total_area: int
    ) -> Tuple[List[Tuple[int, int, int, int]], Optional[Dict]]:
        """
        Identify and analyze regions of significant activity in the heatmap.

        Args:
            heatmap: Normalized heatmap array
            frame_number: Current frame number
            timestamp: Current video timestamp in seconds
            total_area: Total frame area in pixels

        Returns:
            Tuple containing:
            - List of rectangle coordinates (x, y, width, height)
            - Optional dictionary with analysis data for the frame
        """
        # Create binary mask using activity threshold
        threshold = np.mean(heatmap) + self.sensitivity * np.std(heatmap)
        binary_mask = (heatmap > threshold).astype(np.uint8) * 255

        # Find contours in binary mask
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Get valid rectangles
        orig_width, orig_height = binary_mask.shape
        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                rectangles.append((x, y, w, h))
                # this is an old validation to see if area is atleast 1% of the total area
                # if w * h * 100 / total_area > 0.9:

        # Find largest non-contained rectangle
        largest_rect = None
        largest_area = 0
        for rect in rectangles:
            x, y, w, h = rect
            if w * h > largest_area:
                largest_area = w * h
                largest_rect = rect

        # Create analysis data if significant activity found
        analysis_data = None
        if largest_rect and largest_area * 100 / total_area > 1:
            x, y, w, h = largest_rect

            if w > 0.6 * orig_width:
                # Ignore large rectangles (likely full-screen activity)
                return rectangles, None

            analysis_data = {
                'frame': frame_number,
                'timestamp': timestamp,
                'time_str': self.format_timestamp(timestamp),
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'focus': True,
                'area_percentage': largest_area * 100 / total_area,
                'activity_roi_count': len(rectangles)
            }

        return rectangles, analysis_data

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Format timestamp in seconds to HH:MM:SS.mm format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def process_video(
        self,
        video_path: str,
        interval: float = 0.5,
        save_heatmaps: bool = False
    ) -> pd.DataFrame:
        """
        Process video file and generate activity analysis.

        Args:
            video_path: Path to input video file
            interval: Time interval between analyzed frames in seconds
            save_heatmaps: Whether to save heatmap visualizations for each frame

        Returns:
            DataFrame containing analysis results

        Raises:
            ValueError: If video file cannot be opened
        """
        logger.info(f"Starting video processing: {video_path}")

        # Create heatmap subdirectory if needed
        if save_heatmaps:
            self.heatmap_subdir.mkdir(exist_ok=True)
            logger.info(
                f"Heatmap visualizations will be saved to: {self.heatmap_subdir}")

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps * interval)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize results storage
        results_df = pd.DataFrame(columns=[
            'frame', 'timestamp', 'time_str', 'x', 'y', 'width', 'height',
            'focus', 'area_percentage', 'activity_roi_count'
        ])

        frame_count = 1
        prev_frame = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    if prev_frame is not None:
                        # Calculate current timestamp
                        current_time = frame_count / fps

                        # Process frame pair
                        magnitude, _, _ = self.compute_optical_flow(
                            prev_frame, frame)
                        heatmap = self.create_heatmap(magnitude)

                        # Analyze activity regions
                        total_area = frame.shape[0] * frame.shape[1]
                        current_frame_number = frame_count // frame_interval
                        rectangles, analysis_data = self.find_activity_regions(
                            heatmap,
                            current_frame_number,
                            current_time,
                            total_area
                        )

                        # Save heatmap visualization if requested and significant activity found
                        if save_heatmaps and analysis_data is not None:
                            self.save_heatmap_visualization(
                                heatmap,
                                rectangles,
                                current_frame_number,
                                current_time
                            )

                        if analysis_data:
                            results_df.loc[len(results_df)] = analysis_data

                    prev_frame = frame.copy()

                # Log progress
                if frame_count % fps == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Processing progress: {progress:.1f}%")

                frame_count += 1

        finally:
            cap.release()

        # Save results
        output_path = self.output_dir / 'heatmap_results.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"Analysis complete. Results saved to: {output_path}")

        return results_df


def main():
    """Main function to demonstrate usage of VideoHeatmapAnalyzer."""
    analyzer = VideoHeatmapAnalyzer(
        output_dir='heatmap_outputs',
        min_area=100,
        sensitivity=0.5
    )

    video_path = "video.mp4"
    try:
        # Process video with heatmap visualization saving enabled
        results = analyzer.process_video(
            video_path,
            interval=0.5,
            save_heatmaps=True
        )
        print(f"Processed {len(results)} frames with significant activity")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise


if __name__ == "__main__":
    main()
