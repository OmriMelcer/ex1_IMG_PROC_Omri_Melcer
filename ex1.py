import numpy as np
import mediapy as media
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def compute_frame_cdf(frame):
    """
    Convert frame to grayscale, compute normalized histogram, and return CDF.
    
    :param frame: RGB frame from video
    :return: cumulative distribution function (CDF) of the grayscale histogram
    """
    # Convert to grayscale
    gray = rgb2gray(frame)
    
    # Compute normalized histogram (256 bins for grayscale)
    hist, _ = np.histogram(gray, bins=256, range=(0, 1))
    hist = hist / hist.sum()  # normalize to sum = 1
    
    # Compute cumulative histogram (CDF)
    cdf = np.cumsum(hist)
    
    return cdf


def detect_scene_change(video_path, video_type=1):
    """
    Detect scene change in a video with exactly two scenes.
    
    :param video_path: path to video file
    :param video_type: category of video (1 or 2) - used for visualization only
    :return: frame index where scene change occurs (last frame of first scene)
    """
    # Read video frames
    frames = media.read_video(video_path)
    num_frames = len(frames)
    
    # Compute CDF for each frame
    cdfs = []
    for frame in frames:
        cdf = compute_frame_cdf(frame)
        cdfs.append(cdf)
    
    # Compute L¹ distance between consecutive CDFs
    # L¹ norm is robust to both clean cuts and gradual color changes
    distances = []
    for i in range(1, num_frames):
        distance = np.sum(np.abs(cdfs[i] - cdfs[i-1]))
        distances.append(distance)
    
    # Scene change occurs at frame with maximum distance
    max_idx = np.argmax(distances)
    
    return max_idx


def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    scene_change_frame = detect_scene_change(video_path, video_type)
    
    # Return tuple: (last frame of first scene, first frame of second scene)
    return (scene_change_frame, scene_change_frame + 1)


def visualize_scene_change(video_path, scene_change_frame, video_type=1, output_prefix="result"):
    """
    Visualize the scene change detection results.
    Shows frames before/after the cut and the distance curve.
    
    :param video_path: path to video file
    :param scene_change_frame: frame index where scene change occurs
    :param video_type: category of video (affects distance metric)
    :param output_prefix: prefix for saved visualization files
    """
    # Read video frames
    frames = media.read_video(video_path)
    num_frames = len(frames)
    
    # Compute CDFs and distances (matching the detection logic)
    cdfs = []
    for frame in frames:
        cdf = compute_frame_cdf(frame)
        cdfs.append(cdf)
    
    # Compute L¹ distances (same as detection logic)
    distances = []
    for i in range(1, num_frames):
        distance = np.sum(np.abs(cdfs[i] - cdfs[i-1]))
        distances.append(distance)
    
    metric_name = "L¹ Distance"
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Distance curve
    plt.subplot(2, 3, 1)
    plt.plot(distances, linewidth=2)
    plt.axvline(x=scene_change_frame, color='r', linestyle='--', linewidth=2, label='Scene Change')
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'CDF {metric_name} Between Consecutive Frames', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show frames around the scene change
    frames_to_show = [
        (scene_change_frame - 2, "2 frames before cut"),
        (scene_change_frame - 1, "1 frame before cut"),
        (scene_change_frame, "Last frame of Scene 1"),
        (scene_change_frame + 1, "First frame of Scene 2"),
        (scene_change_frame + 2, "2 frames after cut"),
    ]
    
    for idx, (frame_idx, label) in enumerate(frames_to_show, start=2):
        if 0 <= frame_idx < num_frames:
            plt.subplot(2, 3, idx)
            plt.imshow(frames[frame_idx])
            plt.title(f"{label}\n(Frame {frame_idx})", fontsize=11, fontweight='bold')
            plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization (don't show)
    video_name = video_path.split('/')[-1].replace('.mp4', '')
    output_file = f"{output_prefix}_{video_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved: {output_file}")
    
    # Close without showing
    plt.close()


def test_on_exercise_inputs():
    """
    Test the scene change detection on all videos in the Exercise Inputs directory.
    """
    import os
    
    # Path to exercise inputs directory
    exercise_dir = "Exercise Inputs-20251101"
    
    # Get all video files
    video_files = [
        ("video1_category1.mp4", 1),
        ("video2_category1.mp4", 1),
        ("video3_category2.mp4", 2),
        ("video4_category2.mp4", 2)
    ]
    
    print("=" * 60)
    print("Scene Change Detection Results")
    print("=" * 60)
    
    for video_name, category in video_files:
        video_path = os.path.join(exercise_dir, video_name)
        
        if os.path.exists(video_path):
            print(f"\nProcessing: {video_name} (Category {category})")
            try:
                last_frame_scene1, first_frame_scene2 = main(video_path, category)
                print(f"  Scene change detected:")
                print(f"    - Last frame of Scene 1: {last_frame_scene1}")
                print(f"    - First frame of Scene 2: {first_frame_scene2}")
                
                # Create visualization
                visualize_scene_change(video_path, last_frame_scene1, category)
                
            except Exception as e:
                print(f"  Error processing video: {e}")
        else:
            print(f"\nVideo not found: {video_path}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_on_exercise_inputs()
