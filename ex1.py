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
    return (int(scene_change_frame), int(scene_change_frame + 1))


def analyze_histogram_transformation(video_path, video_type=1):
    """
    Analyze the histogram transformation at filter effects to identify
    what kind of manipulation is happening (gamma, brightness, contrast, etc.)
    
    :param video_path: path to video file
    :param video_type: category of video
    """
    # Read video frames
    frames = media.read_video(video_path)
    num_frames = len(frames)
    
    # Compute CDFs
    cdfs = []
    for frame in frames:
        cdf = compute_frame_cdf(frame)
        cdfs.append(cdf)
    
    # Compute L¹ distances
    l1_distances = []
    for i in range(1, num_frames):
        distance = np.sum(np.abs(cdfs[i] - cdfs[i-1]))
        l1_distances.append(distance)
    
    # Find the second largest peak (filter effect)
    l1_sorted_indices = np.argsort(l1_distances)[::-1]
    scene_change_idx = l1_sorted_indices[0]
    filter_effect_idx = l1_sorted_indices[1]
    
    # Get the actual frames for transformation analysis
    frame_before = frames[filter_effect_idx]
    frame_after = frames[filter_effect_idx + 1]
    
    # Convert to grayscale
    gray_before = rgb2gray(frame_before)
    gray_after = rgb2gray(frame_after)
    
    # Flatten to get all pixel values
    pixels_before = gray_before.flatten()
    pixels_after = gray_after.flatten()
    
    # Create a mapping by binning
    bins = np.linspace(0, 1, 257)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # For each input intensity, find the average output intensity
    mapping = np.zeros(256)
    counts = np.zeros(256)
    
    for i in range(len(pixels_before)):
        bin_idx = int(pixels_before[i] * 255)
        if bin_idx >= 256:
            bin_idx = 255
        mapping[bin_idx] += pixels_after[i]
        counts[bin_idx] += 1
    
    # Average the mappings
    for i in range(256):
        if counts[i] > 0:
            mapping[i] = mapping[i] / counts[i]
        else:
            # Interpolate missing values
            mapping[i] = i / 255.0
    
    # Smooth the mapping to see the trend
    from scipy.ndimage import gaussian_filter1d
    mapping_smooth = gaussian_filter1d(mapping, sigma=3)
    
    # Test different transformations
    x = np.linspace(0, 1, 256)
    
    # Test for histogram equalization
    # In histogram equalization, the transformation is the CDF of the input
    hist_before, _ = np.histogram(gray_before, bins=256, range=(0, 1))
    hist_before_norm = hist_before / hist_before.sum()
    cdf_before = np.cumsum(hist_before_norm)
    
    # The histogram equalization transformation should map input to its CDF
    histeq_error = np.mean((mapping_smooth - cdf_before) ** 2)
    
    # Test gamma values
    gamma_candidates = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5]
    gamma_errors = []
    for gamma in gamma_candidates:
        predicted = x ** gamma
        error = np.mean((predicted - mapping_smooth) ** 2)
        gamma_errors.append(error)
    
    best_gamma_idx = np.argmin(gamma_errors)
    best_gamma = gamma_candidates[best_gamma_idx]
    
    # Test linear transformations
    brightness_shift = np.mean(mapping_smooth - x)
    contrast_scale = np.polyfit(x - 0.5, mapping_smooth - 0.5, 1)[0]
    
    # Check if output histogram is more uniform
    hist_after, _ = np.histogram(gray_after, bins=256, range=(0, 1))
    hist_after_norm = hist_after / hist_after.sum()
    
    # Measure uniformity using standard deviation (lower = more uniform)
    uniform_hist = np.ones(256) / 256
    uniformity_before = np.std(hist_before_norm)
    uniformity_after = np.std(hist_after_norm)
    ideal_uniformity = np.std(uniform_hist)
    
    # Chi-square test for uniformity
    chi2_before = np.sum((hist_before_norm - uniform_hist) ** 2 / (uniform_hist + 1e-10))
    chi2_after = np.sum((hist_after_norm - uniform_hist) ** 2 / (uniform_hist + 1e-10))
    
    print(f"\n  Transformation Analysis for Filter at frame {filter_effect_idx} → {filter_effect_idx+1}:")
    print(f"    Histogram Equalization fit: MSE = {histeq_error:.6f}")
    print(f"    Best fitting gamma: γ = {best_gamma:.2f} (MSE: {gamma_errors[best_gamma_idx]:.6f})")
    print(f"    Brightness shift: {brightness_shift:+.4f}")
    print(f"    Contrast scale: {contrast_scale:.4f}")
    print(f"\n    Histogram Uniformity Analysis:")
    print(f"      Std Dev before: {uniformity_before:.6f}")
    print(f"      Std Dev after:  {uniformity_after:.6f} (lower = more uniform)")
    print(f"      Ideal uniform:  {ideal_uniformity:.6f}")
    print(f"      Uniformity improvement: {(uniformity_before - uniformity_after) / uniformity_before * 100:.1f}%")
    
    # Determine the most likely transformation
    if histeq_error < 0.01 and uniformity_after < uniformity_before * 0.7:
        print(f"    → Likely HISTOGRAM EQUALIZATION (spreading intensities uniformly)")
    elif uniformity_after < uniformity_before * 0.8:
        print(f"    → Likely histogram STRETCHING or partial equalization")
    elif gamma_errors[best_gamma_idx] < 0.001:
        print(f"    → Likely a GAMMA CORRECTION with γ ≈ {best_gamma}")
    elif abs(brightness_shift) > 0.05 and contrast_scale < 1.2:
        print(f"    → Likely a BRIGHTNESS adjustment ({brightness_shift:+.2f})")
    elif abs(contrast_scale - 1.0) > 0.2:
        print(f"    → Likely a CONTRAST adjustment (scale: {contrast_scale:.2f})")
    else:
        print(f"    → Complex transformation (combination or other effect)")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    
    # Row 1: Frames
    plt.subplot(3, 4, 1)
    plt.imshow(frame_before)
    plt.title(f'Frame {filter_effect_idx} (Before Filter)', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(frame_after)
    plt.title(f'Frame {filter_effect_idx+1} (After Filter)', fontweight='bold')
    plt.axis('off')
    
    # Row 1: Grayscale versions
    plt.subplot(3, 4, 3)
    plt.imshow(gray_before, cmap='gray')
    plt.title('Grayscale Before', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(gray_after, cmap='gray')
    plt.title('Grayscale After', fontweight='bold')
    plt.axis('off')
    
    # Row 2: Transformation curve
    plt.subplot(3, 4, 5)
    plt.scatter(pixels_before[::100], pixels_after[::100], alpha=0.1, s=1, c='blue', label='Pixel mapping')
    plt.plot(x, mapping_smooth, 'r-', linewidth=3, label='Smooth mapping', alpha=0.8)
    plt.plot(x, x, 'k--', linewidth=1, label='Identity (no change)', alpha=0.5)
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title('Intensity Transformation Function', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Row 2: Test different gamma values
    plt.subplot(3, 4, 6)
    plt.plot(x, mapping_smooth, 'r-', linewidth=3, label='Actual', alpha=0.8)
    plt.plot(x, cdf_before, 'purple', linewidth=2.5, linestyle='--', label='Hist Equalization')
    for gamma in [0.7, 1.0, 1.3, 2.0]:
        plt.plot(x, x ** gamma, '--', alpha=0.5, linewidth=1, label=f'γ={gamma}')
    plt.plot(x, x ** best_gamma, 'g-', linewidth=2, label=f'Best γ={best_gamma}')
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title('Transformation Comparison', fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Row 2: Linear transformations
    plt.subplot(3, 4, 7)
    plt.plot(x, mapping_smooth, 'r-', linewidth=3, label='Actual', alpha=0.8)
    plt.plot(x, x + brightness_shift, 'b--', linewidth=2, label=f'Brightness: +{brightness_shift:.3f}')
    plt.plot(x, contrast_scale * (x - 0.5) + 0.5, 'g--', linewidth=2, label=f'Contrast: ×{contrast_scale:.2f}')
    plt.plot(x, x, 'k--', linewidth=1, alpha=0.5)
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.title('Linear Transformation Comparison', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Row 2: Error analysis
    plt.subplot(3, 4, 8)
    plt.bar(range(len(gamma_candidates)), gamma_errors)
    plt.xticks(range(len(gamma_candidates)), [f'{g:.1f}' for g in gamma_candidates])
    plt.xlabel('Gamma Value')
    plt.ylabel('Mean Squared Error')
    plt.title('Gamma Fitting Errors', fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Row 3: Histograms
    hist_before, _ = np.histogram(gray_before, bins=256, range=(0, 1))
    hist_after, _ = np.histogram(gray_after, bins=256, range=(0, 1))
    hist_before = hist_before / hist_before.sum()
    hist_after = hist_after / hist_after.sum()
    uniform_ref = np.ones(256) / 256
    
    plt.subplot(3, 4, 9)
    plt.plot(x, hist_before, label='Before', linewidth=2, alpha=0.7, color='blue')
    plt.plot(x, hist_after, label='After', linewidth=2, alpha=0.7, color='orange')
    plt.axhline(y=1/256, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Uniform')
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram Comparison', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Row 3: CDFs
    cdf_before = np.cumsum(hist_before)
    cdf_after = np.cumsum(hist_after)
    
    plt.subplot(3, 4, 10)
    plt.plot(x, cdf_before, label='Before', linewidth=2, alpha=0.7)
    plt.plot(x, cdf_after, label='After', linewidth=2, alpha=0.7)
    plt.xlabel('Intensity')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF Comparison', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Row 3: Difference plots
    plt.subplot(3, 4, 11)
    plt.plot(x, hist_after - hist_before, linewidth=2, color='purple')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Intensity')
    plt.ylabel('Difference')
    plt.title('Histogram Difference', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 12)
    plt.plot(x, cdf_after - cdf_before, linewidth=2, color='orange')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Intensity')
    plt.ylabel('Difference')
    plt.title('CDF Difference', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    video_name = video_path.split('/')[-1].replace('.mp4', '')
    output_file = f"transformation_analysis_{video_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Transformation analysis saved: {output_file}")
    plt.close()


def analyze_filter_effect(video_path, video_type=1):
    """
    Analyze the filter effect by comparing histograms and CDFs of frames
    around the largest and second-largest distance peaks.
    
    :param video_path: path to video file
    :param video_type: category of video
    """
    # Read video frames
    frames = media.read_video(video_path)
    num_frames = len(frames)
    
    # Compute histograms and CDFs
    hists = []
    cdfs = []
    for frame in frames:
        gray = rgb2gray(frame)
        hist, _ = np.histogram(gray, bins=256, range=(0, 1))
        hist = hist / hist.sum()
        hists.append(hist)
        cdfs.append(np.cumsum(hist))
    
    # Compute both L¹ and L∞ distances
    l1_distances = []
    linf_distances = []
    for i in range(1, num_frames):
        l1_dist = np.sum(np.abs(cdfs[i] - cdfs[i-1]))
        linf_dist = np.max(np.abs(cdfs[i] - cdfs[i-1]))
        l1_distances.append(l1_dist)
        linf_distances.append(linf_dist)
    
    # Find the two largest peaks
    l1_sorted_indices = np.argsort(l1_distances)[::-1]
    linf_sorted_indices = np.argsort(linf_distances)[::-1]
    
    scene_change_idx = l1_sorted_indices[0]
    filter_effect_idx_l1 = l1_sorted_indices[1]
    filter_effect_idx_linf = linf_sorted_indices[1]
    
    print(f"\n  Analysis for understanding filter effects:")
    print(f"    Largest L¹ peak (scene change): frame {scene_change_idx} → {scene_change_idx+1}, distance = {l1_distances[scene_change_idx]:.4f}")
    print(f"    2nd largest L¹ peak (filter?): frame {filter_effect_idx_l1} → {filter_effect_idx_l1+1}, distance = {l1_distances[filter_effect_idx_l1]:.4f}")
    print(f"    2nd largest L∞ peak (filter?): frame {filter_effect_idx_linf} → {filter_effect_idx_linf+1}, distance = {linf_distances[filter_effect_idx_linf]:.4f}")
    
    # Create detailed visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: L¹ Distance curve
    plt.subplot(3, 4, 1)
    plt.plot(l1_distances, linewidth=2, label='L¹', color='blue')
    plt.axvline(x=scene_change_idx, color='red', linestyle='--', linewidth=2, label='Scene Change')
    plt.axvline(x=filter_effect_idx_l1, color='orange', linestyle='--', linewidth=2, label='2nd Peak (Filter?)')
    plt.xlabel('Frame Index')
    plt.ylabel('L¹ Distance')
    plt.title('L¹ Distance Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: L∞ Distance curve
    plt.subplot(3, 4, 2)
    plt.plot(linf_distances, linewidth=2, label='L∞', color='green')
    plt.axvline(x=scene_change_idx, color='red', linestyle='--', linewidth=2, label='Scene Change')
    plt.axvline(x=filter_effect_idx_linf, color='purple', linestyle='--', linewidth=2, label='2nd Peak (Filter?)')
    plt.xlabel('Frame Index')
    plt.ylabel('L∞ Distance')
    plt.title('L∞ Distance Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Analyze scene change (row 1, columns 3-4)
    plt.subplot(3, 4, 3)
    plt.plot(hists[scene_change_idx], label=f'Frame {scene_change_idx}', linewidth=2, alpha=0.7)
    plt.plot(hists[scene_change_idx + 1], label=f'Frame {scene_change_idx+1}', linewidth=2, alpha=0.7)
    plt.xlabel('Intensity Bin')
    plt.ylabel('Normalized Frequency')
    plt.title('Histograms: Scene Change', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 4)
    plt.plot(cdfs[scene_change_idx], label=f'Frame {scene_change_idx}', linewidth=2, alpha=0.7)
    plt.plot(cdfs[scene_change_idx + 1], label=f'Frame {scene_change_idx+1}', linewidth=2, alpha=0.7)
    plt.xlabel('Intensity Bin')
    plt.ylabel('Cumulative Probability')
    plt.title('CDFs: Scene Change', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Analyze L¹ filter effect (row 2, columns 1-2)
    plt.subplot(3, 4, 5)
    plt.imshow(frames[filter_effect_idx_l1])
    plt.title(f'L¹ Filter: Frame {filter_effect_idx_l1}', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(frames[filter_effect_idx_l1 + 1])
    plt.title(f'L¹ Filter: Frame {filter_effect_idx_l1+1}', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.plot(hists[filter_effect_idx_l1], label=f'Frame {filter_effect_idx_l1}', linewidth=2, alpha=0.7)
    plt.plot(hists[filter_effect_idx_l1 + 1], label=f'Frame {filter_effect_idx_l1+1}', linewidth=2, alpha=0.7)
    plt.xlabel('Intensity Bin')
    plt.ylabel('Normalized Frequency')
    plt.title('Histograms: L¹ 2nd Peak (Filter Effect)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 8)
    plt.plot(cdfs[filter_effect_idx_l1], label=f'Frame {filter_effect_idx_l1}', linewidth=2, alpha=0.7)
    plt.plot(cdfs[filter_effect_idx_l1 + 1], label=f'Frame {filter_effect_idx_l1+1}', linewidth=2, alpha=0.7)
    plt.xlabel('Intensity Bin')
    plt.ylabel('Cumulative Probability')
    plt.title('CDFs: L¹ 2nd Peak (Filter Effect)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Analyze L∞ filter effect (row 3, columns 1-2)
    plt.subplot(3, 4, 9)
    plt.imshow(frames[filter_effect_idx_linf])
    plt.title(f'L∞ Filter: Frame {filter_effect_idx_linf}', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(frames[filter_effect_idx_linf + 1])
    plt.title(f'L∞ Filter: Frame {filter_effect_idx_linf+1}', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.plot(hists[filter_effect_idx_linf], label=f'Frame {filter_effect_idx_linf}', linewidth=2, alpha=0.7)
    plt.plot(hists[filter_effect_idx_linf + 1], label=f'Frame {filter_effect_idx_linf+1}', linewidth=2, alpha=0.7)
    plt.xlabel('Intensity Bin')
    plt.ylabel('Normalized Frequency')
    plt.title('Histograms: L∞ 2nd Peak (Filter Effect)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 12)
    plt.plot(cdfs[filter_effect_idx_linf], label=f'Frame {filter_effect_idx_linf}', linewidth=2, alpha=0.7)
    plt.plot(cdfs[filter_effect_idx_linf + 1], label=f'Frame {filter_effect_idx_linf+1}', linewidth=2, alpha=0.7)
    plt.xlabel('Intensity Bin')
    plt.ylabel('Cumulative Probability')
    plt.title('CDFs: L∞ 2nd Peak (Filter Effect)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save analysis
    video_name = video_path.split('/')[-1].replace('.mp4', '')
    output_file = f"filter_analysis_{video_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Filter analysis saved: {output_file}")
    plt.close()


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
                
                # Analyze filter effects (especially for category 2)
                analyze_filter_effect(video_path, category)
                
                # Analyze the histogram transformation
                analyze_histogram_transformation(video_path, category)
                
            except Exception as e:
                print(f"  Error processing video: {e}")
        else:
            print(f"\nVideo not found: {video_path}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_on_exercise_inputs()
