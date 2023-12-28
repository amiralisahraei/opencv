import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


# Read and show an image
def read_image(image_path):
    try:
        image_path = os.path.join(image_path)
        myimage = cv.imread(image_path)

        return myimage

    except Exception as err:
        print("There is something wrong:", err)


def stitiching(image_paths_to_stitch):
    # Retrieve image files
    image_files = glob.glob(image_paths_to_stitch)
    image_files = sorted(image_files)

    # Create array of images
    images = []

    for file in image_files:
        image = read_image(file)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        images.append(image)

    images_len = len(images)

    # Plot the images
    num_columns = 3
    num_rows = int(np.ceil(images_len / num_columns))

    for i in range(images_len):
        plt.subplot(num_rows + 1, num_columns, i + 1)
        plt.axis("off")
        plt.title(f"Image {i+1}")
        plt.imshow(images[i])

    # Stitch Images
    stitcher = cv.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)

    if status == 0:
        plt.subplot(num_rows + 1, 1, num_rows + 1)
        plt.axis("off")
        plt.title("Stitched Image")
        plt.imshow(stitched_image)

    plt.show()


collection_path = "./images/*"
stitiching(collection_path)
