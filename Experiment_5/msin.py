import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from pathlib import Path



# STEP 1: GET CURRENT SCRIPT FOLDER


BASE_DIR = Path(__file__).resolve().parent

image1_path = BASE_DIR / "img5A.jpg"
image2_path = BASE_DIR / "img5B.jpg"

print("Working Folder:")
print(BASE_DIR)

print("\nImage 1:")
print(image1_path)

print("\nImage 2:")
print(image2_path)



# STEP 2: CHECK IMAGE FILES


if not image1_path.exists():
    raise FileNotFoundError(
        f"image1.jpg not found!\nExpected location: {image1_path}"
    )

if not image2_path.exists():
    raise FileNotFoundError(
        f"image2.jpg not found!\nExpected location: {image2_path}"
    )



# STEP 3: LOAD IMAGES


image1 = cv2.imread(str(image1_path))
image2 = cv2.imread(str(image2_path))

if image1 is None:
    raise ValueError(
        f"OpenCV could not read image1.jpg.\n"
        f"File exists at: {image1_path}\n"
        f"The image may be corrupted or not a valid JPG file."
    )

if image2 is None:
    raise ValueError(
        f"OpenCV could not read image2.jpg.\n"
        f"File exists at: {image2_path}\n"
        f"The image may be corrupted or not a valid JPG file."
    )

print("\nImages loaded successfully!")



# STEP 4: CONVERT BGR TO RGB


image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)



# STEP 5: CONVERT TO GRAYSCALE


gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)



# STEP 6: SIFT FEATURE EXTRACTION


sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

print("\n==============================")
print("SIFT FEATURE EXTRACTION")
print("==============================")

print("Image 1 Keypoints:", len(keypoints1))
print("Image 2 Keypoints:", len(keypoints2))



# STEP 7: VISUALIZE SIFT KEYPOINTS


sift_image1 = cv2.drawKeypoints(
    image1_rgb,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

sift_image2 = cv2.drawKeypoints(
    image2_rgb,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(sift_image1)
plt.title("SIFT Keypoints - Image 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sift_image2)
plt.title("SIFT Keypoints - Image 2")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "sift_keypoints.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# STEP 8: HOG FEATURE EXTRACTION


hog_features, hog_image = hog(
    gray1,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True
)

print("\n==============================")
print("HOG FEATURE EXTRACTION")
print("==============================")

print("Number of HOG Features:", len(hog_features))



# STEP 9: HOG VISUALIZATION


hog_image_rescaled = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray1, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Visualization")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    BASE_DIR / "hog_visualization.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# STEP 10: SIFT IMAGE MATCHING


print("\n==============================")
print("SIFT IMAGE MATCHING")
print("==============================")


if descriptors1 is not None and descriptors2 is not None:

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    good_matches = []

    for match in matches:

        if len(match) == 2:

            m, n = match

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print("Total Matches:", len(matches))
    print("Good Matches:", len(good_matches))


    # =====================================================
    # STEP 11: DISPLAY MATCHES
    # =====================================================

    matched_image = cv2.drawMatches(
        image1_rgb,
        keypoints1,
        image2_rgb,
        keypoints2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(15, 8))

    plt.imshow(matched_image)

    plt.title("SIFT Feature Matching")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        BASE_DIR / "sift_matching.jpg",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

else:

    print("Descriptors could not be calculated.")



# STEP 12: SAVE HOG FEATURES


np.save(
    BASE_DIR / "hog_features.npy",
    hog_features
)



# STEP 13: FINAL OUTPUT


print("\n================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("================================")

print("\nGenerated Files:")

print("1. sift_keypoints.jpg")
print("2. hog_visualization.jpg")
print("3. sift_matching.jpg")
print("4. hog_features.npy")