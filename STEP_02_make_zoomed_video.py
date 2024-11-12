"""
Video Processing Module for Zoom Effects and Clip Management
This module provides functionality for creating various zoom effects on video clips
and managing video compositions using MoviePy.
"""

from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips


def create_zoom_in_effect(video_path: str, output_path: str, focus_point_x: int,
                          focus_point_y: int, start_time: float, end_time: float) -> None:
    """
    Creates a zoom-in effect on a video clip, focusing on a specific point.

    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where the processed video will be saved
        focus_point_x (int): X-coordinate of the zoom focus point
        focus_point_y (int): Y-coordinate of the zoom focus point
        start_time (float): Start time of the zoom effect in seconds
        end_time (float): End time of the zoom effect in seconds
    """
    duration = end_time - start_time
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)

    # Get original dimensions
    width, height = clip.size

    def calculate_zoom_parameters(time):
        """Calculate zoom parameters for each frame."""
        zoom_factor = 1 + 0.5 * (time/duration)
        new_width = width * zoom_factor
        new_height = height * zoom_factor

        # Maintain focus point position
        position_x = -(focus_point_x * zoom_factor - focus_point_x)
        position_y = -(focus_point_y * zoom_factor - focus_point_y)

        return {
            'width': new_width,
            'height': new_height,
            'position': (position_x, position_y)
        }

    # Apply zoom effect
    zoomed_clip = (clip
                   .resize(lambda t: (calculate_zoom_parameters(t)['width'],
                                      calculate_zoom_parameters(t)['height']))
                   .set_position(lambda t: calculate_zoom_parameters(t)['position'])
                   .set_duration(duration))

    # Create and export final composition
    final_composition = CompositeVideoClip([zoomed_clip], size=clip.size)
    final_composition.write_videofile(output_path,
                                      fps=video.fps,
                                      codec='libx264',
                                      audio_codec='aac')

    # Clean up resources
    for clip_obj in [video, clip, zoomed_clip, final_composition]:
        clip_obj.close()


def create_zoom_out_effect(video_path: str, output_path: str, focus_point_x: int,
                           focus_point_y: int, start_time: float, end_time: float) -> None:
    """
    Creates a zoom-out effect on a video clip, starting from a focused point.

    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where the processed video will be saved
        focus_point_x (int): X-coordinate of the zoom focus point
        focus_point_y (int): Y-coordinate of the zoom focus point
        start_time (float): Start time of the zoom effect in seconds
        end_time (float): End time of the zoom effect in seconds
    """
    duration = end_time - start_time
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)

    # Get original dimensions
    width, height = clip.size

    def calculate_zoom_parameters(time):
        """Calculate zoom parameters for each frame."""
        zoom_factor = 1 + 0.5 * (1 - time/duration)  # Changed 't' to 'time'
        new_width = width * zoom_factor
        new_height = height * zoom_factor

        # Maintain focus point position
        position_x = -(focus_point_x * zoom_factor - focus_point_x)
        position_y = -(focus_point_y * zoom_factor - focus_point_y)

        return {
            'width': new_width,
            'height': new_height,
            'position': (position_x, position_y)
        }

    # Apply zoom effect
    zoomed_clip = (clip
                   .resize(lambda t: (calculate_zoom_parameters(t)['width'],
                                      calculate_zoom_parameters(t)['height']))
                   .set_position(lambda t: calculate_zoom_parameters(t)['position'])
                   .set_duration(duration))

    # Create and export final composition
    final_composition = CompositeVideoClip([zoomed_clip], size=clip.size)
    final_composition.write_videofile(output_path,
                                      fps=video.fps,
                                      codec='libx264',
                                      audio_codec='aac')

    # Clean up resources
    for clip_obj in [video, clip, zoomed_clip, final_composition]:
        clip_obj.close()


def apply_static_zoom(video_path: str, output_path: str, focus_point_x: int,
                      focus_point_y: int, start_time: float, end_time: float) -> None:
    """
    Applies a constant zoom level to a video segment.

    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where the processed video will be saved
        focus_point_x (int): X-coordinate of the zoom focus point
        focus_point_y (int): Y-coordinate of the zoom focus point
        start_time (float): Start time of the zoom effect in seconds
        end_time (float): End time of the zoom effect in seconds
    """
    # Initialize video and create subclip
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)

    # Get original dimensions
    width, height = clip.size

    # Define constant zoom parameters
    zoom_factor = 1.5  # 150% zoom
    new_width = width * zoom_factor
    new_height = height * zoom_factor

    # Calculate position to maintain focus point
    position_x = -(focus_point_x * zoom_factor - focus_point_x)
    position_y = -(focus_point_y * zoom_factor - focus_point_y)

    # Create zoomed clip with static parameters
    zoomed_clip = (clip
                   .resize((new_width, new_height))
                   .set_position((position_x, position_y))
                   .set_duration(end_time - start_time))

    # Create final composition
    final_composition = CompositeVideoClip([zoomed_clip], size=clip.size)

    # Export the video
    final_composition.write_videofile(output_path,
                                      fps=video.fps,
                                      codec='libx264',
                                      audio_codec='aac')

    # Clean up resources
    for clip_obj in [video, clip, zoomed_clip, final_composition]:
        clip_obj.close()


def concatenate_video_segments(output_path: str, clip_paths: list) -> None:
    """
    Combines multiple video segments into a single video file.

    Args:
        output_path (str): Path where the final merged video will be saved
        clip_paths (list): List of paths to video segments to be merged
    """
    clips = [VideoFileClip(path) for path in clip_paths]
    final_clip = concatenate_videoclips(clips, method='compose')

    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Clean up resources
    for clip in clips + [final_clip]:
        clip.close()


if __name__ == '__main__':
    # Configuration
    FOCUS_POINT = {
        'x': 1154,
        'y': 686
    }

    TIME_SEGMENTS = {
        'zoom_in': (5, 10),
        'static_zoom': (10, 12),
        'zoom_out': (12, 17)
    }

    INPUT_VIDEO = '/Users/bishwaraj/Documents/screencast-refiner/composite_videos/fdebd853-a815-46ad-bbd2-335471d2d640_composite.mp4'

    # Process video segments
    create_zoom_in_effect(INPUT_VIDEO, 'segment1.mp4',
                          FOCUS_POINT['x'], FOCUS_POINT['y'],
                          *TIME_SEGMENTS['zoom_in'])

    apply_static_zoom(INPUT_VIDEO, 'segment2.mp4',
                      FOCUS_POINT['x'], FOCUS_POINT['y'],
                      *TIME_SEGMENTS['static_zoom'])

    create_zoom_out_effect(INPUT_VIDEO, 'segment3.mp4',
                           FOCUS_POINT['x'], FOCUS_POINT['y'],
                           *TIME_SEGMENTS['zoom_out'])

    # Merge all segments
    concatenate_video_segments('final_output.mp4',
                               ['segment1.mp4', 'segment2.mp4', 'segment3.mp4'])
