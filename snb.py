import datetime
import random
import time
import urllib.request
from urllib.request import urlopen
import logging
import argparse

import cv2  # type: ignore
import numpy as np # type: ignore
from yamager import Yamager # type: ignore

from augmentation import crop_transformations, rotate_transformations, scale_transformations, shear_transformations
from augmentation import translate_transformations, b_contrast, addNoise_gaussian, addNoise_salt_pepper, addNoise_poisson

yamager = Yamager()


# The function of getting a random image through the Google search engine
def get_image_url():
    images_url = yamager.search_google_images(search)
    return random.choice(images_url)


# The function of getting a random User Agent to simulate a natural request
def get_user_agents(filename):
    file_user_agents = open(filename, 'r')
    user_agents_list = file_user_agents.read()
    return user_agents_list.split('\n')


# The function of getting a list of queries for the search engine
def get_search_queries(filename):
    file_search_queries = open(filename, 'r')
    search_list = file_search_queries.read()
    return search_list.split('\n')


# The function of converting the size of the final image to 640x480 and saving
def image_save(image, path, filename, quality, size):
    image_out = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
    write_status = cv2.imwrite(path + '/' + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if write_status is True:
        return True
    else:
        return False


# Image acquisition function
def get_image(url, user_agent):
    req = urllib.request.Request(url, data=None, headers={'User-Agent': user_agent})
    req = urlopen(req)
    image = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image


# We get the current date
current_date = datetime.datetime.now().strftime('%d%m%Y')

# We get a list of User Agents (2000 variants)
user_agents_list = get_user_agents('user_agents.txt')

# We get a list of search queries to search for images on a given topic
search_list = get_search_queries('search_queries.txt')

# Saving logs
logging.basicConfig(
    level=logging.INFO,
    filename="get_images.log",
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    datefmt='%d.%m.%Y - %H:%M:%S',
)


###############################################################################
# Program parameters
###############################################################################

# Getting a list with startup parameters
def get_console_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", help="Image counter")
    parser.add_argument("--aug", help="Augmentation: True / False")
    parser.add_argument("--quality", help="Quality (0-100%)")
    parser.add_argument("--folder", help="Save folder")

    args = parser.parse_args()
    c = int(args.count) - 1
    a = str(args.aug)
    q = int(args.quality)
    f = str(args.folder)

    params = list()

    params.append(c)
    params.append(a)
    params.append(q)
    params.append(f)
    return params


params_list = get_console_params()

# Enabling / disabling augmentation
get_augmentation = params_list[1]

# The size parameter of the saved image (the bilinear interpolation method is used to adjust the size)
size = [640, 480]

image_count = params_list[0]

# The quality parameter of the saved image (0-100%)
quality = params_list[2]

# Save folder
folder = params_list[3]

###############################################################################

iteration = 0
count = 0
count_all_images = 0
number_augment_modif_all = 0

while count <= image_count:
    number_augment_modif = 0

    # We get a random search query
    search = search_list[random.randint(0, len(search_list) - 1)]
    print('Request to the search engine: ' + search)

    try:
        url = get_image_url()
        print('The image address is being processed: ' + url)

        user_agent = user_agents_list[random.randint(0, len(user_agents_list) - 1)]

        # We get the original image
        image = get_image(url, user_agent)
        if (image_save(image, folder, str(iteration) + str(current_date) + '.jpg', quality, size)):
            count += 1
            print('Original images uploaded: ', count)
            print('---------------------------------------------')

            # Block of various transformations (augmentation)
            if (get_augmentation == "true"):
                image_save(crop_transformations(image), folder, 'crop_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1
                image_save(rotate_transformations(image), folder,
                           'rotate_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1
                image_save(scale_transformations(image), folder, 'scale_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1
                image_save(shear_transformations(image), folder, 'shear_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1
                image_save(translate_transformations(image), folder,
                           'translate_' + str(iteration) + str(current_date) + '.jpg', quality, size)
                number_augment_modif += 1

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_save(b_contrast(image), folder, 'b_co_' + str(iteration) + str(current_date) + '.jpg', quality,
                           size)
                number_augment_modif += 1
                image_save(b_contrast(image, 0.7, -3), folder, 'b_c_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1
                image_save(addNoise_gaussian(image), folder, 'gaussian_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1
                image_save(addNoise_salt_pepper(image), folder, 'pepper_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1
                image_save(addNoise_poisson(image), folder, 'poisson_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1

                # Switching the color space to shades of gray
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_save(b_contrast(image), folder, 'b_contrast_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1
                image_save(b_contrast(image, 0.7, -10), folder, 'b_cont_' + str(iteration) + str(current_date) + '.jpg',
                           quality, size)
                number_augment_modif += 1

        number_augment_modif_all += number_augment_modif
        count_all_images = count + number_augment_modif_all
        # Writing a log
        logging.info('The serial number of the image: ' + str(count))
        logging.info('URL: ' + url)
        logging.info('Number of modifications (augmentation): ' + str(number_augment_modif))
        logging.info('The resulting number of images in the set: ' + str(count_all_images))
        print('The resulting number of images in the set: ' + str(count_all_images))
        logging.info('--------------------------------------------------------------------')
        print('---------------------------------------------')

        # To simulate real queries, we set a random pause time
        time_pause = random.randint(1, 25)
        time.sleep(time_pause)

        iteration += 1

    except Exception:
        continue
