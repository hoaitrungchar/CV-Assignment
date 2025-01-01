import cv2
import os
from typing import Union, Optional

def extract_frames(
    video_path: str,
    output_dir: str,
    prefix: str = 'frame',
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    frame_interval: int = 1,
    max_frames: Optional[int] = None
) -> int:
    """
    Extract frames from a video file with advanced selection options.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        prefix: Prefix for frame filenames
        start_time: Start time in seconds (None for beginning)
        end_time: End time in seconds (None for entire video)
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to extract (None for all)
    
    Returns:
        Number of frames extracted
    
    Raises:
        ValueError: If invalid parameters are provided
        IOError: If video file cannot be opened
    """
    if not os.path.exists(video_path):
        raise IOError(f"Video file not found: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Set video position based on start_time
    if start_time is not None:
        if not 0 <= start_time < duration:
            raise ValueError(f"Invalid start_time: {start_time}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
    
    if end_time is not None:
        if not start_time < end_time <= duration:
            raise ValueError(f"Invalid end_time: {end_time}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or (end_time and frame_count/fps >= end_time):
            break
            
        if frame_count % frame_interval == 0:
            if max_frames and saved_count >= max_frames:
                break
                
            frame_filename = os.path.join(output_dir, f"{prefix}_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Extracted {saved_count} frames...")
                
        frame_count += 1
    
    cap.release()
    return saved_count

if __name__ == "__main__":
    video_file = 'input_video.mp4'
    output_folder = 'extracted_frames'
    
    try:
        frames = extract_frames(
            video_file,
            output_folder,
            start_time=0,      # Start from beginning
            end_time=None,     # Process until end
            frame_interval=1,  # Extract every frame
            max_frames=None    # No limit
        )
        print(f"Successfully extracted {frames} frames to {output_folder}")
    except Exception as e:
        print(f"Error: {str(e)}")