import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import urllib
import sys


# Download the other models here
# https://ashwin-phadke.github.io/post/load-tensorflow-models-using-opencv/
# Necessary paths
modelFile = "models/mobileNet/frozen_inference_graph.pb"
configFile = "models/mobileNet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"

# Make sure "models" directory has been created
if not os.path.isdir("models"):
    os.mkdir("models")

# Make sure the model file has been downloaded
if not os.path.isfile(modelFile):
    # Go inside "models" directory
    os.chdir("models")

    # Download the model file
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
        "ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
    )

    # Come back to the previous directory
    os.chdir("..")

# Read the Class Labels
with open(classFile) as fp:
    labels = fp.read().split("\n")

# Read the model
model = cv.dnn.readNetFromTensorflow(modelFile, configFile)


# Retrieve detected objects
def detect_object(model, image):
    dim = 300

    # Preprocessong step
    blob = cv.dnn.blobFromImage(
        image, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False
    )

    # Pass the image to the model
    model.setInput(blob)

    # Perform Prediction
    detected_objects = model.forward()

    return detected_objects


# Display the label of the corresponding object
def display_label(image, text, x, y):
    # Get text size
    labelsize, baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
    # Draw rectangle
    cv.rectangle(
        image,
        (x, y - int(1.2 * labelsize[1])),
        (x + labelsize[0], y),
        (0, 0, 0),
        cv.FILLED,
    )

    cv.putText(
        image,
        text,
        (x, y - baseline),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )


# Display the bounding box
def display_object(image, detected_objects, threshold=0.25):
    image_width = image.shape[1]
    image_height = image.shape[0]

    for i in range(detected_objects.shape[2]):
        # Find the class and confidence
        classId = int(detected_objects[0, 0, i, 1])
        score = float(detected_objects[0, 0, i, 2])

        # Coordinates to draw bounding box
        x = int(detected_objects[0, 0, i, 3] * image_width)
        y = int(detected_objects[0, 0, i, 4] * image_height)
        w = int(detected_objects[0, 0, i, 5] * image_width)
        h = int(detected_objects[0, 0, i, 6] * image_height)

        if score > threshold:
            display_label(image, f"{labels[classId]}", x, y)
            cv.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

    return image


# Test the model
def test_image(test_image_path):
    test_image = cv.imread(test_image_path)
    detected_objects = detect_object(model, test_image)
    final_image = display_object(test_image, detected_objects)
    final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)
    return final_image


def display_test_images(images_directory_path):
    test_images_file = os.listdir(images_directory_path)[0:4]
    num_rows = int(np.ceil(len(test_images_file) / 2))
    plt.figure(figsize=(15, 15))
    for i in range(len(test_images_file)):
        image_file = test_image("./images/" + test_images_file[i])
        plt.subplot(num_rows, 2, i + 1)
        plt.imshow(image_file)
        plt.axis("off")

    plt.show()


# display_test_images("./images")


# Apply Object detection for Camera
def open_camera():
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    source = cv.VideoCapture(s)

    win_name = "Camera Preview"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    while cv.waitKey(1) != ord("q"):  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        frame = cv.flip(frame, 1)
        detected_objects = detect_object(model, frame)
        result = display_object(frame, detected_objects, 0.7)

        cv.imshow(win_name, result)

    source.release()
    cv.destroyAllWindows()


open_camera()
