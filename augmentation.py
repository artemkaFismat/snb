from augment.geometric import crop, rotate, scale, shear, translate # type: ignore
from augment.photometric import brightness_contrast, colorSpace, addNoise # type: ignore


# Geometric transformation functions
##############################################################################

# Image cropping function
def crop_transformations(image):
    return crop(image, point1=(100, 100), point2=(450, 400))


# Image rotation function
def rotate_transformations(image):
    return rotate(image, angle=15, keep_resolution=True)


# Image zoom function
def scale_transformations(image):
    return scale(image, fx=1.5, fy=1.5, keep_resolution=False)


# Shift function with image rotation
def shear_transformations(image):
    return shear(image, shear_val=0.2, axis=1)


# Image movement function
def translate_transformations(image):
    return translate(image, tx=50, ty=60)


# Photometric transformation functions
##############################################################################

# Photometric transformation functions
def b_contrast(image, alpha=1.3, beta=5):
    return brightness_contrast(image, alpha, beta)


# Colorspace change function = 'hsv', 'ycrcb' 'lab'
def color_space(image, colorspace='hsv'):
    return colorSpace(image, colorspace)


# Gaussian noise reduction function
def addNoise_gaussian(image, mean=0, var=0.08):
    return addNoise(image, 'gaussian', mean, var)


# The function of making noise "pepper"
def addNoise_salt_pepper(image, sp_ratio=0.5, noise_amount=0.1):
    return addNoise(image, 'salt_pepper', sp_ratio=sp_ratio, noise_amount=noise_amount)


# Poisson noise reduction function
def addNoise_poisson(image, noise_amount=0.5):
    return addNoise(image, 'poisson', noise_amount=noise_amount)
