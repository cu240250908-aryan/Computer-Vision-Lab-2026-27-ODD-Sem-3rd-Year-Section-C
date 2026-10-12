import cv2
import numpy as np
import os

# EXPERIMENT 8
# Optical Flow Based Object Tracking

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "optical_flow_output.mp4"

# Open Input Video

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("ERROR: input.mp4 not found!")
    print("Please put input.mp4 in the same folder as main.py")
    exit()

# Video information
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("========================================")
print("Video opened successfully")
print("Width  :", width)
print("Height :", height)
print("FPS    :", fps)
print("========================================")

# Create Output Video

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video!")
    cap.release()
    exit()

# Lucas-Kanade Parameters

feature_params = dict(
    maxCorners=200,
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

# Random colors
colors = np.random.randint(
    0, 255,
    (200, 3)
)

# Read First Frame

ret, old_frame = cap.read()

if not ret:
    print("ERROR: Could not read first frame!")
    cap.release()
    out.release()
    exit()

old_gray = cv2.cvtColor(
    old_frame,
    cv2.COLOR_BGR2GRAY
)

# Shi-Tomasi Corner Detection
p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)

# Trajectory mask
mask = np.zeros_like(old_frame)

frame_count = 0

# PROCESS VIDEO

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # LUCAS-KANADE OPTICAL FLOW

    if p0 is not None:

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            p0,
            None,
            **lk_params
        )

        if p1 is not None:

            good_new = p1[status == 1]
            good_old = p0[status == 1]

            for i, (new, old) in enumerate(
                zip(good_new, good_old)
            ):

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
                    colors[i % 200].tolist(),
                    2
                )

                # Draw feature point
                frame = cv2.circle(
                    frame,
                    (x_new, y_new),
                    4,
                    colors[i % 200].tolist(),
                    -1
                )

                # Draw motion vector
                cv2.arrowedLine(
                    frame,
                    (x_old, y_old),
                    (x_new, y_new),
                    (0, 255, 0),
                    1,
                    tipLength=0.3
                )

            p0 = good_new.reshape(-1, 1, 2)

    # FARNEBACK DENSE OPTICAL FLOW

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

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    average_motion = np.mean(magnitude)
    average_direction = np.mean(angle) * 180 / np.pi

    # COMBINE FRAME + TRAJECTORY

    output_frame = cv2.add(
        frame,
        mask
    )

    # DISPLAY INFORMATION

    cv2.putText(
        output_frame,
        "OPTICAL FLOW OBJECT TRACKING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output_frame,
        f"Motion Magnitude: {average_motion:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    cv2.putText(
        output_frame,
        f"Direction: {average_direction:.2f} degrees",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    cv2.putText(
        output_frame,
        f"Frame: {frame_count}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # SAVE EVERY FRAME

    out.write(output_frame)

    # Show output
    cv2.imshow(
        "Optical Flow Tracking",
        output_frame
    )

    # UPDATE PREVIOUS FRAME

    old_gray = frame_gray.copy()

    frame_count += 1

    # Re-detect features every 30 frames
    if frame_count % 30 == 0:

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )

        mask = np.zeros_like(old_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Video processing stopped by user.")
        break


# RELEASE EVERYTHING

cap.release()
out.release()
cv2.destroyAllWindows()

# CHECK OUTPUT FILE

if os.path.exists(OUTPUT_VIDEO):

    file_size = os.path.getsize(
        OUTPUT_VIDEO
    ) / (1024 * 1024)

    print("\n========================================")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("========================================")
    print("Total Frames :", frame_count)
    print("Output File  :", OUTPUT_VIDEO)
    print(f"File Size    : {file_size:.2f} MB")
    print("========================================")
    print("Output video has been saved successfully!")

else:

    print("\nERROR: Output video was not created.")