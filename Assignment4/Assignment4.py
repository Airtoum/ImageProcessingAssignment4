import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy import ndimage

static = imread("Image_20449.tif")
moving = imread("Image_20450.tif")

static_thr = static > 350
moving_thr = moving > 350

#fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=False)
#ax1.imshow(static_thr)
#ax2.imshow(moving_thr)

static_particlesx = []
static_particlesy = []
moving_particlesx = []
moving_particlesy = []
for y in range(static_thr.shape[0] - 1):
    for x in range(static_thr.shape[1] - 1):
        if static_thr[y,x]:
            static_particlesx.append(x)
            static_particlesy.append(y)
for y in range(moving_thr.shape[0] - 1):
    for x in range(moving_thr.shape[1] - 1):
        if moving_thr[y,x]:
            moving_particlesx.append(x)
            moving_particlesy.append(y)
static_center = (sum(static_particlesy) / len(static_particlesy), sum(static_particlesx) / len(static_particlesx)) 
moving_center = (sum(moving_particlesy) / len(moving_particlesy), sum(moving_particlesx) / len(moving_particlesx)) 
#the centers are in format (y,x) since that's how the image is stored
print(static_center)
#static_center = static_center[::-1]
#moving_center = moving_center[::-1]

fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=False)
fig.suptitle("centers of mass")

ax1.imshow(static)
ax1.set_title("static")
ax1.scatter(static_center[1], static_center[0], c='red', marker='o')
# plotting the centers needs to be swapped since the plot is in (x,y)
ax2.imshow(moving)
ax2.set_title("moving")
ax2.scatter(moving_center[1], moving_center[0], c='red', marker='o')

plt.show()


displacement = (static_center[0] - moving_center[0], static_center[1] - moving_center[1])
moving_center = (  moving_center[0] + displacement[0], moving_center[1] + displacement[1]  )
assert moving_center[0] == static_center[0]
assert moving_center[1] == static_center[1]
moving = ndimage.shift(moving, displacement)

fig, ((ax1, ax2)) = plt.subplots(1, 2, sharex=False, sharey=False)
fig.suptitle("aligned centers of mass")

ax1.imshow(static)
ax1.set_title("static")
ax1.scatter(static_center[1], static_center[0], c='red', marker='o')

ax2.imshow(moving)
ax2.set_title("moving")
ax2.scatter(moving_center[1], moving_center[0], c='red', marker='o')

plt.show()


def SSD(a, b):
    return np.sum((np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32))**2)

def rotate(array, center, degrees):
    #https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python/25459080
    padX = np.array([array.shape[1] - center[1], center[1]]).astype(int)
    padY = np.array([array.shape[0] - center[0], center[0]]).astype(int)
    padded = np.pad(array, [padY, padX], "constant")
    rotated = ndimage.rotate(padded, degrees, reshape=False)
    cropped = rotated[ padY[0] : -padY[1], padX[0] : -padX[1] ]
    return cropped

'''
def rotateImage(img, angle, pivot):
    padX = np.rint( [img.shape[1] - pivot[0], pivot[0]] )
    padY = np.rint[img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
'''

theta_to_SSD = {}
lowest_SSD = np.finfo(np.float32).max
best_theta = None
best_image = None
for theta in range(0, 360, 5):
    print("Testing theta = " + str(theta))
    moving_temp = rotate(moving, moving_center, theta)
    current_SSD = SSD(static, moving_temp)
    theta_to_SSD[theta] = current_SSD
    if current_SSD < lowest_SSD:
        lowest_SSD = current_SSD
        best_theta = theta
        best_image = moving_temp
        print("New best")
    
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False)
fig.suptitle("Rotation Iteration 1")
ax1.imshow(static)
ax1.set_title("static")
ax1.scatter(static_center[1], static_center[0], c='red', marker='o')
ax2.imshow(best_image)
#ax2.imshow(rotateImage(moving, 80, moving_center[::-1]))
ax2.set_title("moving")
ax2.scatter(moving_center[1], moving_center[0], c='red', marker='o')
ax3.scatter(theta_to_SSD.keys(), theta_to_SSD.values())
ax3.set_title("SSD plot")
print("Lowest log(SSD) is " + str(np.log10(lowest_SSD)))
plt.show()

root_theta = best_theta
for doubleoffset in range(-10, 10):
    theta = root_theta + doubleoffset / 2.0
    print("Testing theta = " + str(theta))
    moving_temp = rotate(moving, moving_center, theta)
    current_SSD = SSD(static, moving_temp)
    theta_to_SSD[theta] = current_SSD
    if current_SSD < lowest_SSD:
        lowest_SSD = current_SSD
        best_theta = theta
        best_image = moving_temp
        print("New best")

moving = best_image

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False)
fig.suptitle("Rotation Iteration 2")
ax1.imshow(static)
ax1.set_title("static")
ax1.scatter(static_center[1], static_center[0], c='red', marker='o')
ax2.imshow(moving)
ax2.set_title("moving")
ax2.scatter(moving_center[1], moving_center[0], c='red', marker='o')
ax3.scatter(theta_to_SSD.keys(), theta_to_SSD.values())
ax3.set_title("SSD plot")
print("Lowest log(SSD) is " + str(np.log10(lowest_SSD)))
plt.show()

yx_to_SSD = {}
lowest_SSD = np.finfo(np.float32).max
best_yx = None
best_image = None
translation_step = 10
translation_range_in_steps = 5  # total of (5*2 + 1)^2 = 121
for s_y in range(-translation_range_in_steps, translation_range_in_steps):
    t_y = s_y * translation_step
    for s_x in range(-translation_range_in_steps, translation_range_in_steps):
        t_x = s_x * translation_step
        yx = (int(t_y), int(t_x))
        print("Testing (y,x) = " + str(yx) )
        moving_temp = ndimage.shift(moving, yx)
        current_SSD = SSD(static, moving_temp)
        yx_to_SSD[yx] = current_SSD
        if current_SSD < lowest_SSD:
            lowest_SSD = current_SSD
            best_yx = yx
            best_image = moving_temp
            print("New best")

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False)
fig.suptitle("Translation Iteration 1")
ax1.imshow(static)
ax1.set_title("static")
ax2.imshow(best_image)
ax2.set_title("moving")
ax3.scatter(np.array(list(yx_to_SSD.keys()))[:,0], np.array(list(yx_to_SSD.keys()))[:,1], c=list(yx_to_SSD.values()))
ax3.set_title("SSD plot")
print("Lowest log(SSD) is " + str(np.log10(lowest_SSD)))
plt.show()

root_yx = best_yx
translation_step = 1 # no reason to translate less than one pixel
translation_range_in_steps = 5  # total of (5*2 + 1)^2 = 121
for s_y in range(-translation_range_in_steps, translation_range_in_steps):
    t_y = s_y * translation_step + root_yx[0]
    for s_x in range(-translation_range_in_steps, translation_range_in_steps):
        t_x = s_x * translation_step + root_yx[1]
        yx = (int(t_y), int(t_x))
        print("Testing (y,x) = " + str(yx) )
        moving_temp = ndimage.shift(moving, yx)
        current_SSD = SSD(static, moving_temp)
        yx_to_SSD[yx] = current_SSD
        if current_SSD < lowest_SSD:
            lowest_SSD = current_SSD
            best_yx = yx
            best_image = moving_temp
            print("New best")

moving = best_image
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False)
fig.suptitle("Translation Iteration 2")
ax1.imshow(static)
ax1.set_title("static")
ax2.imshow(moving)
ax2.set_title("moving")
ax3.scatter(np.array(list(yx_to_SSD.keys()))[:,0], np.array(list(yx_to_SSD.keys()))[:,1], c=list(yx_to_SSD.values()))
ax3.set_title("SSD plot")
print("Lowest log(SSD) is " + str(np.log10(lowest_SSD)))
plt.show()