import os
import cv2 as cv
import matplotlib.pyplot as plt
import urllib


# Make sure "model" directory has been created
if not os.path.isdir("model"):
    os.mkdir("model")

# The path of the model file
protoFile = "model/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "model/pose_iter_160000.caffemodel"

# Make sure the necessary file has been created
if not os.path.isfile(protoFile):
    # Download the proto file
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt",
        protoFile,
    )

if not os.path.isfile(weightsFile):
    # Download the model file
    urllib.request.urlretrieve(
        "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel",
        weightsFile,
    )


# Pose points of the different parst of body
# For example: 0 => head  1 => neck
number_points = 15
POSE_PAIRS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 14],
    [14, 8],
    [8, 9],
    [9, 10],
    [14, 11],
    [11, 12],
    [12, 13],
]

# Read the model
model = cv.dnn.readNetFromCaffe(protoFile, weightsFile)


# Read the test image
def test_image_info(test_image_path):
    test_image = cv.imread(test_image_path)
    test_image = cv.cvtColor(test_image, cv.COLOR_BGR2RGB)
    test_image_width = test_image.shape[1]
    test_image_height = test_image.shape[0]

    return (test_image, test_image_width, test_image_height)


test_image, test_image_width, test_image_height = test_image_info("./images/Tiger_Woods_crop.png")

# Preprocessing
model_input_size = (368, 368)
blob = cv.dnn.blobFromImage(
    test_image, 1.0 / 255, model_input_size, (0, 0, 0), swapRB=True, crop=False
)


# Run the model on test image
model.setInput(blob)
output = model.forward()


# Display probablity maps for keypoints
def display_probabliy():
    for i in range(number_points):
        prob_map = output[0, i, :, :]
        displayMap = cv.resize(
            prob_map, (test_image_width, test_image_height), cv.INTER_LINEAR
        )
        plt.subplot(2, 8, i + 1)
        plt.axis("off")
        plt.imshow(displayMap, cmap="jet")
        plt.show()


# display_probabliy()


# Extract Keypoints
def extract_keypoints(
    output, number_points, test_image_width, test_image_height, threshold
):
    scaleX = test_image_width / output.shape[3]
    scaleY = test_image_height / output.shape[2]

    # Empty list to store the detected keypoints
    points = []

    for i in range(number_points):
        # Obtain probablity map
        prob_map = output[0, i, :, :]

        # Find global maxima of the probMap
        minval, prob, minLoc, point = cv.minMaxLoc(prob_map)

        # Scale the keypoint coordiantes to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold:
            points.append((int(x), int(y)))
        else:
            points.append(None)

    return points


threshold = 0.1
points = extract_keypoints(
    output, number_points, test_image_width, test_image_height, threshold
)


# Display Points & Skeleton
image_Points = test_image.copy()
image_Skeleton = test_image.copy()


# Points
def display_only_keypoints(points, image_Points):
    for i, p in enumerate(points):
        cv.circle(image_Points, p, 8, (0, 255, 0), thickness=-1, lineType=cv.FILLED)
        cv.putText(
            image_Points,
            f"{i}",
            p,
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            lineType=cv.LINE_AA,
        )

    return image_Points


# Skeleton
def display_skeleton(pose_pairs, points):
    for pair in pose_pairs:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv.line(
                image_Skeleton,
                points[partA],
                points[partB],
                (255, 255, 0),
                2,
            )
            cv.circle(
                image_Skeleton,
                points[partA],
                8,
                (255, 0, 0),
                thickness=-1,
                lineType=cv.FILLED,
            )
    return image_Skeleton


plt.subplot(1, 2, 1)
plt.imshow(display_only_keypoints(points, image_Points))
plt.axis("off")
plt.title("Keypoints")

plt.subplot(1, 2, 2)
plt.imshow(display_skeleton(POSE_PAIRS, points))
plt.axis("off")
plt.title("Skeleton")

plt.show()
