#### Overview

The `VideoHeatmapAnalyzer` class is designed to analyze video content and generate activity heatmaps to identify regions of significant activity within screen recordings. This tool is particularly useful for applications where understanding spatial distribution and intensity of activity over time is crucial, such as enhancing user experience by automatically zooming into active areas of screen recordings.

#### Features

- Computes optical flow between video frames to detect motion.
- Generates heatmaps based on motion magnitudes.
- Identifies and analyzes regions with significant activities.
- Optionally saves visualizations of these heatmaps with activity regions highlighted.

#### Algorithm

The analysis process involves several steps:

1. **Initialization**: Configure output directories and settings for area thresholds and sensitivity.

2. **Optical Flow Calculation**:

   - Convert frames to grayscale.
   - Create a margin mask to ignore edges of the frame.
   - Use the Farneback method to compute dense optical flow.
   - Convert flow vectors to polar coordinates (magnitude and angle) and apply the margin mask.

3. **Heatmap Creation**:

   - Apply Gaussian smoothing to the magnitude of optical flow to create a smoothed heatmap.
   - Normalize the heatmap for better visibility and thresholding.

4. **Activity Region Identification**:

   - Apply a threshold based on the mean and standard deviation of the heatmap values to create a binary mask.
   - Detect contours in the binary mask and qualify them based on size criteria.
   - Optionally filter out large regions that likely represent less meaningful activity.

5. **Results Processing**:

   - For each frame processed, save heatmap images if enabled.

   <a href="https://ibb.co/kyqQTFx"><img src="https://i.ibb.co/FW8g1tK/heatmap-frame-14-6-72s.png" alt="heatmap-frame-14-6-72s" border="0"></a>

   - Gather data on activity regions including their dimensions and location, and compile this information into a DataFrame.

6. **Final Output**:

   - Save the DataFrame as a CSV file containing all significant activity instances with timestamps, coordinates, and other pertinent information.

| frame | timestamp | time_str     | x    | y   | width | height | focus | area_percentage    | activity_roi_count |
| ----- | --------- | ------------ | ---- | --- | ----- | ------ | ----- | ------------------ | ------------------ |
| 11    | 5.28      | 00:00:05.280 | 1138 | 727 | 440   | 85     | True  | 2.012310606060606  | 1                  |
| 14    | 6.72      | 00:00:06.720 | 1090 | 354 | 513   | 450    | True  | 12.420906508264462 | 2                  |
| 15    | 7.2       | 00:00:07.200 | 1121 | 329 | 374   | 450    | True  | 9.055397727272727  | 45                 |
| 17    | 8.16      | 00:00:08.160 | 1154 | 687 | 405   | 87     | True  | 1.8958225723140496 | 5                  |
| 18    | 8.64      | 00:00:08.640 | 1142 | 350 | 419   | 398    | True  | 8.972645488980715  | 4                  |
| 20    | 9.6       | 00:00:09.600 | 1558 | 517 | 178   | 304    | True  | 2.911501377410468  | 1                  |
| 21    | 10.08     | 00:00:10.080 | 1531 | 443 | 206   | 191    | True  | 2.11701532369146   | 23                 |
| 26    | 12.48     | 00:00:12.480 | 1195 | 589 | 315   | 76     | True  | 1.2880940082644627 | 3                  |

#### Environment Setup

Before running the script, ensure the following setup steps are completed:

1. **Poetry Installation**:

   - Ensure you have Poetry installed on your system as it is used for dependency management.

2. **Dependency Setup**:

   - Run `poetry install` to install dependencies and set up the environment.

3. **Virtual Environment**:

   - Start the project environment using `poetry shell` to activate the `.venv`.

4. **Library Management**:

   - To add new libraries, use `poetry add <package>`.
   - To remove libraries, use `poetry remove <package>`.

#### Usage

To use the `VideoHeatmapAnalyzer`, follow these steps:

1. Instantiate the analyzer:

   ```python
   analyzer = VideoHeatmapAnalyzer(output_dir='desired_output_directory', min_area=100, sensitivity=0.5)
   ```

2. Call the `process_video` method with the path to your video:

   ```python
   results = analyzer.process_video(video_path='path_to_your_video.mp4', interval=0.5, save_heatmaps=True)
   ```

3. Results will be saved in the specified output directory, and significant activity frames will be printed to the console.

#### Example

Here's a simple example to get you started:

```python
def main():
    analyzer = VideoHeatmapAnalyzer(output_dir='heatmap_outputs', min_area=100, sensitivity=0.5)
    video_path = "path_to_video.mp4"
    try:
        results = analyzer.process_video(video_path, interval=0.5, save_heatmaps=True)
        print(f"Processed {len(results)} frames with significant activity.")
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
```
