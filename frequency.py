import cv2
import numpy as np
import matplotlib.pyplot as plt

f = cv2.imread('photos/11.jpg')
f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)  

magnitude_spectrum = np.abs(Fshift)

def radial_profile(data, center):
    y, x = np.indices(data.shape)  
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)  
    r = r.astype(int)  

    radial_sum = np.bincount(r.ravel(), data.ravel())  
    radial_count = np.bincount(r.ravel())  
    radial_profile = radial_sum / radial_count 

    return radial_profile

center = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
radial_profile_data = radial_profile(magnitude_spectrum, center)

peak_frequency = np.argmax(radial_profile_data)

print(f"Peak frequency in the radial profile is at radius: {radial_profile_data}")

peak_found = 0
for i in range(1, len(radial_profile_data) - 1):  
    diff = radial_profile_data[i+1] - radial_profile_data[i]  
    print(i, " - ", radial_profile_data[i])
    if diff > 5e+05:  
        peak_found = 1
        x = radial_profile_data[i]

print(peak_found, ":", i, " ", radial_profile_data[i])
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.plot(radial_profile_data)
plt.title('Radial Profile')
plt.xlabel('Frequency Radius')
plt.ylabel('Average Intensity')
plt.grid()

plt.tight_layout()
plt.show()
