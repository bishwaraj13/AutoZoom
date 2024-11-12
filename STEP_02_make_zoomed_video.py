from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
import pandas as pd
import numpy as np


def process_video(video_path, csv_path, output_path):
    # Read the video and CSV
    video = VideoFileClip(video_path)
    df = pd.read_csv(csv_path)
    fps = video.fps

    # Convert frame numbers to timestamps
    df['timestamp'] = df['frame'] / fps

    # Sort by timestamp to ensure proper sequence
    df = df.sort_values('timestamp')

    # Create subclips list
    subclips = []

    # Process each focus point
    for i in range(len(df)):
        row = df.iloc[i]

        # Calculate start and end times
        start_time = row['timestamp']
        if i < len(df) - 1:
            end_time = df.iloc[i + 1]['timestamp']
        else:
            end_time = video.duration

        # Create subclip
        current_clip = video.subclip(start_time, end_time)

        # Calculate zoom center coordinates
        x, y = row['x'], row['y']

        # Create zoom effect
        zoom_duration = end_time - start_time
        zoom_in_duration = min(0.5, zoom_duration / 3)  # Quick zoom in
        steady_duration = zoom_duration - zoom_in_duration * 2  # Hold the zoom

        def get_zoom_factor(t):
            if t < zoom_in_duration:
                # Zoom in effect
                return 1 + 0.5 * (t / zoom_in_duration)
            elif t < zoom_in_duration + steady_duration:
                # Hold zoom
                return 1.5
            else:
                # Zoom out effect
                t_out = t - (zoom_in_duration + steady_duration)
                return 1.5 - 0.5 * (t_out / zoom_in_duration)

        # Apply zoom effect
        zoomed_clip = (current_clip
                       .resize(lambda t: get_zoom_factor(t))
                       .set_position(lambda t: (
                           'center' if t < zoom_in_duration else
                           (
                               max(0, min(1, (video.w/2 - x)/(video.w/2))),
                               max(0, min(1, (video.h/2 - y)/(video.h/2)))
                           )
                       ))
                       .set_duration(zoom_duration))

        # Create composite clip
        final_clip = CompositeVideoClip(
            [zoomed_clip],
            size=video.size
        )

        subclips.append(final_clip)

    # Concatenate all subclips
    final_video = concatenate_videoclips(subclips)

    # Write final video
    final_video.write_videofile(
        output_path,
        fps=fps,
        codec='libx264',
        audio_codec='aac'
    )

    # Clean up
    video.close()
    final_video.close()
    for clip in subclips:
        clip.close()


input_video = "/Users/bishwaraj/Documents/screencast-refiner/composite_videos/fdebd853-a815-46ad-bbd2-335471d2d640_composite.mp4"
output_video = "kkk.mp4"


process_video(input_video, "heatmap_results_old.csv", output_video)
