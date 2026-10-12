import cv2
import numpy as np
import os

# EXPERIMENT 7
# Motion Estimation using Optical Flow
# Lucas-Kanade Sparse + Farneback Dense Optical Flow

# Input and output video
input_video = "sample-5s.mp4"
output_video = "output_optical_flow.mp4"

# Check input video

if not os.path.exists(input_video):
    print("ERROR: Input video not found!")
    print("Make sure the video is in the same folder as main.py")
    print("Required file:", input_video)
    exit()

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video opened successfully!")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total Frames:", total_frames)

# Output video

# Output will contain two videos side by side
output_width = width * 2
output_height = height

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (output_width, output_height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit()

# Read first frame

ret, old_frame = cap.read()

if not ret:
    print("ERROR: Could not read first frame.")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Lucas-Kanade parameters

feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# Detect feature points
p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

# Mask for drawing trajectories
mask = np.zeros_like(old_frame)

frame_number = 1

# PROCESS VIDEO

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ========================================================
    # LUCAS-KANADE SPARSE OPTICAL FLOW
    # ========================================================

    sparse_frame = frame.copy()

    if p0 is not None and len(p0) > 0:

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            **lk_params
        )

        if p1 is not None:

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for new, old in zip(good_new, good_old):

                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()

                x_new = int(x_new)
                y_new = int(y_new)

                x_old = int(x_old)
                y_old = int(y_old)

                # Draw trajectory
                mask = cv2.line(
                    mask,
                    (x_new, y_new),
                    (x_old, y_old),
                    (255, 255, 255),
                    2
                )

                # Draw motion arrow
                sparse_frame = cv2.arrowedLine(
                    sparse_frame,
                    (x_old, y_old),
                    (x_new, y_new),
                    (0, 255, 0),
                    2,
                    tipLength=0.3
                )

                # Draw feature point
                sparse_frame = cv2.circle(
                    sparse_frame,
                    (x_new, y_new),
                    4,
                    (0, 0, 255),
                    -1
                )

            # Update tracking points
            if len(good_new) > 0:
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = cv2.goodFeaturesToTrack(
                    frame_gray,
                    mask=None,
                    **feature_params
                )

    # Add trajectories
    sparse_frame = cv2.add(
        sparse_frame,
        mask
    )

    # ========================================================
    # FARNEBACK DENSE OPTICAL FLOW
    # ========================================================

    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    # Calculate magnitude and direction
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    # HSV visualization
    hsv = np.zeros_like(frame)

    # Direction
    hsv[..., 0] = angle * 180 / np.pi / 2

    # Saturation
    hsv[..., 1] = 255

    # Motion magnitude
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # Convert HSV to BGR
    dense_frame = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # ========================================================
    # Add titles
    # ========================================================

    cv2.putText(
        sparse_frame,
        "Lucas-Kanade Sparse Optical Flow",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        dense_frame,
        "Farneback Dense Optical Flow",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Frame number
    cv2.putText(
        sparse_frame,
        f"Frame: {frame_number}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # Combine both outputs
    # ========================================================

    combined = np.hstack(
        (sparse_frame, dense_frame)
    )

    # ========================================================
    # SAVE OUTPUT VIDEO
    # ========================================================

    out.write(combined)

    # Show output
    cv2.imshow(
        "Experiment 7 - Optical Flow",
        combined
    )

    # Update previous frame
    old_gray = frame_gray.copy()

    frame_number += 1

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Processing stopped by user.")
        break

# RELEASE

cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("========================================")
print("Experiment 7 completed successfully!")
print("========================================")
print("Output video saved as:")
print(output_video)
print()