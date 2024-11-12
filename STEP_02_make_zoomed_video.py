from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips
import pandas as pd
import numpy as np


# def zoom_clip(video_path, output_path, x, y, start_time, end_time, duration):
#     video = VideoFileClip(video_path)

#     # Extract the portion of the video to zoom in on
#     clip = video.subclip(start_time, end_time)

#     # Create zoomed clip
#     zoomed_clip = (clip
#                    .resize(lambda t: 1 + 0.5*(t/duration))
#                    .set_position((x, y))
#                    .set_duration(duration))

#     # Create composite video
#     composite_video = CompositeVideoClip([zoomed_clip], size=clip.size)

#     # Write video to file
#     composite_video.write_videofile(output_path,
#                                     fps=video.fps,
#                                     codec='libx264',
#                                     audio_codec='aac')

#     # Close the video
#     video.close()
#     clip.close()
#     zoomed_clip.close()
#     composite_video.close()

def zoom_clip(video_path, output_path, x, y, start_time, end_time, duration):
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)

    # Get original dimensions
    w, h = clip.size

    def zoom_effect(t):
        # Calculate zoom factor (1 to 1.5 over duration)
        zoom = 1 + 0.5 * (t/duration)

        # Calculate new dimensions
        new_w = w * zoom
        new_h = h * zoom

        # Calculate position to keep target point (x,y) in same relative position
        pos_x = -(x * zoom - x)
        pos_y = -(y * zoom - y)

        return {
            'w': new_w,
            'h': new_h,
            'pos': (pos_x, pos_y)
        }

    # Create zoomed clip
    zoomed_clip = (clip
                   .resize(lambda t: (zoom_effect(t)['w'], zoom_effect(t)['h']))
                   .set_position(lambda t: zoom_effect(t)['pos'])
                   .set_duration(duration))

    # Create composite video
    final = CompositeVideoClip([zoomed_clip], size=clip.size)

    # Write video to file
    final.write_videofile(output_path,
                          fps=video.fps,
                          codec='libx264',
                          audio_codec='aac')

    # Close all clips
    video.close()
    clip.close()
    zoomed_clip.close()
    final.close()


def revert_zoom(video_path, output_path, x, y, start_time, end_time, duration):
    video = VideoFileClip(video_path)
    clip = video.subclip(start_time, end_time)

    # Get original dimensions
    w, h = clip.size

    def zoom_effect(t):
        # Calculate zoom factor (1.5 to 1 over duration)
        zoom = 1 + 0.5 * (1 - t/duration)

        # Calculate new dimensions
        new_w = w * zoom
        new_h = h * zoom

        # Calculate position to keep target point (x,y) in same relative position
        pos_x = -(x * zoom - x)
        pos_y = -(y * zoom - y)

        return {
            'w': new_w,
            'h': new_h,
            'pos': (pos_x, pos_y)
        }

    # Create zoomed clip
    zoom_reverted_clip = (clip
                          .resize(lambda t: (zoom_effect(t)['w'], zoom_effect(t)['h']))
                          .set_position(lambda t: zoom_effect(t)['pos'])
                          .set_duration(duration))

    # Create composite video
    final = CompositeVideoClip([zoom_reverted_clip], size=clip.size)

    # Write video to file
    final.write_videofile(output_path,
                          fps=video.fps,
                          codec='libx264',
                          audio_codec='aac')

    # Close all clips
    video.close()
    clip.close()
    zoom_reverted_clip.close()
    final.close()


# def revert_zoom(video_path, output_path, x, y, start_time, end_time, duration):
#     video = VideoFileClip(video_path)

#     # Extract the portion of the video to zoom in on
#     clip = video.subclip(start_time, end_time)

#     zoom_reverted_clip = (clip
#                           .resize(lambda t: 1 + 0.5*(1 - t/duration))
#                           .set_position((x, y))
#                           .set_duration(duration))

#     # Create composite video
#     composite_video = CompositeVideoClip([zoom_reverted_clip], size=clip.size)

#     # Write video to file
#     composite_video.write_videofile(output_path,
#                                     fps=video.fps,
#                                     codec='libx264',
#                                     audio_codec='aac')

#     # Close the video
#     video.close()
#     clip.close()
#     zoom_reverted_clip.close()
#     composite_video.close()

# make subclip but it should be in zoomed in state


# def make_constant_zoomed(video_path, x, y, start_time, end_time, output_path):
#     video = VideoFileClip(video_path)
#     subclip = video.subclip(start_time, end_time)

#     # Create subclip which is zoomed in all the time
#     zoomed_clip = (subclip
#                    .resize(1.5)
#                    .set_position((x, y))
#                    .set_duration(end_time - start_time))

#     # Write video to file
#     zoomed_clip.write_videofile(output_path,
#                                 fps=video.fps,
#                                 codec='libx264',
#                                 audio_codec='aac')

#     # Close the video
#     video.close()
#     subclip.close()
#     zoomed_clip.close()

def make_constant_zoomed(video_path, x, y, start_time, end_time, output_path):
    video = VideoFileClip(video_path)
    subclip = video.subclip(start_time, end_time)

    # Get original dimensions
    w, h = subclip.size

    # Calculate constant zoom parameters
    zoom = 1.5
    new_w = w * zoom
    new_h = h * zoom

    # Calculate position to keep target point (x,y) in same relative position
    pos_x = -(x * zoom - x)
    pos_y = -(y * zoom - y)

    # Create zoomed clip
    zoomed_clip = (subclip
                   .resize((new_w, new_h))
                   .set_position((pos_x, pos_y))
                   .set_duration(end_time - start_time))

    # Create composite video with original dimensions
    final = CompositeVideoClip([zoomed_clip], size=subclip.size)

    # Write video to file
    final.write_videofile(output_path,
                          fps=video.fps,
                          codec='libx264',
                          audio_codec='aac')

    # Close all clips
    video.close()
    subclip.close()
    zoomed_clip.close()
    final.close()


def make_subclip(video_path, start_time, end_time, output_path):
    video = VideoFileClip(video_path)
    subclip = video.subclip(start_time, end_time)

    # write to disk
    subclip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the video
    video.close()
    subclip.close()


def merge_clips(output_path):
    clips = [VideoFileClip(clip_path)
             for clip_path in ['ooo.mp4', 'op.mp4', 'ppp.mp4']]
    final_clip = concatenate_videoclips(clips, method='compose')

    # write to disk
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the video
    for clip in clips:
        clip.close()

    final_clip.close()


if __name__ == '__main__':

    # Define zoom parameters
    x = 1154
    y = 686
    start_time = 5
    end_time = 10
    duration = 5

    # Define video path
    video_path = '/Users/bishwaraj/Documents/screencast-refiner/composite_videos/fdebd853-a815-46ad-bbd2-335471d2d640_composite.mp4'
    output_path = 'ooo.mp4'

    # Create zoomed video
    zoom_clip(video_path, output_path, x, y, start_time, end_time, duration)

    output_path = 'ppp.mp4'
    # Revert zoom
    revert_zoom(video_path, output_path, x, y, 12, 17, duration)

    output_path = 'op.mp4'
    make_constant_zoomed(video_path, x, y, 10, 12, output_path)

    # Merge clips
    output_path = 'merged.mp4'
    merge_clips(output_path)
