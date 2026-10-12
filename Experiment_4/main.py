import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load a grayscale image
img = cv2.imread('imge4.jpg', cv2.IMREAD_GRAYSCALE)
# Check if image was loaded successfully
if img is None:
    print("Error: img4.jpg not found!")
    print("Make sure img4.jpg is inside the Experiment_4 folder.")
    exit()

# 2. Compute the Discrete Fourier Transform (DFT)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
# 3. Shift the zero-frequency component to the center
dft_shift = np.fft.fftshift(dft)

# 4. Compute magnitude spectrum
magnitude_spectrum = 20 * np.log(
    cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1
)

# Get image dimensions and center point
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask_size = 30

# 5. Low-Pass Filter (LPF)
mask_lpf = np.zeros((rows, cols, 2), np.uint8)

mask_lpf[
    crow - mask_size:crow + mask_size,
    ccol - mask_size:ccol + mask_size
] = 1

fshift_lpf = dft_shift * mask_lpf

# 6. High-Pass Filter (HPF)
mask_hpf = np.ones((rows, cols, 2), np.uint8)

mask_hpf[
    crow - mask_size:crow + mask_size,
    ccol - mask_size:ccol + mask_size
] = 0

fshift_hpf = dft_shift * mask_hpf

# 7. Inverse Fourier Transform

# LPF Reconstruction
f_ishift_lpf = np.fft.ifftshift(fshift_lpf)
img_back_lpf = cv2.idft(f_ishift_lpf)

img_back_lpf = cv2.magnitude(
    img_back_lpf[:, :, 0],
    img_back_lpf[:, :, 1]
)

# HPF Reconstruction
f_ishift_hpf = np.fft.ifftshift(fshift_hpf)
img_back_hpf = cv2.idft(f_ishift_hpf)

img_back_hpf = cv2.magnitude(
    img_back_hpf[:, :, 0],
    img_back_hpf[:, :, 1]
)

# 8. Normalize images for saving
mag_save = cv2.normalize(
    magnitude_spectrum, None, 0, 255,
    cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

lpf_save = cv2.normalize(
    img_back_lpf, None, 0, 255,
    cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

hpf_save = cv2.normalize(
    img_back_hpf, None, 0, 255,
    cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

# 9. Save output images
cv2.imwrite('magnitude_spectrum.jpg', mag_save)
cv2.imwrite('low_pass_filtered.jpg', lpf_save)
cv2.imwrite('high_pass_filtered.jpg', hpf_save)

print("Images have been saved successfully!")

# 10. Display results
plt.figure(figsize=(12, 10))

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(223)
plt.imshow(img_back_lpf, cmap='gray')
plt.title('Low-Pass Filtered (Smoothed)')
plt.axis('off')

plt.subplot(224)
plt.imshow(img_back_hpf, cmap='gray')
plt.title('High-Pass Filtered (Edges)')
plt.axis('off')

plt.tight_layout()
plt.show()