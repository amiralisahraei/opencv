import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt


# Read and show an image
def read_image(image_path):
    try:
        image_path = os.path.join(image_path)
        myimage = cv.imread(image_path)

        return myimage

    except Exception as err:
        print("There is something wrong:", err)


def registration(form_image_path, scanned_form_image_path):
    im1 = read_image(form_image_path)
    im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)

    im2 = read_image(scanned_form_image_path)
    im2 = cv.cvtColor(im2, cv.COLOR_BGR2RGB)

    # Convert image to grayscal
    im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # ORB features
    MAX_NUM_FEATURES = 800
    orb = cv.ORB_create(MAX_NUM_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Dsipay the keypoints
    im1_display = cv.drawKeypoints(
        im1,
        keypoints1,
        outImage=np.array([]),
        color=(255, 0, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    im2_display = cv.drawKeypoints(
        im2,
        keypoints2,
        outImage=np.array([]),
        color=(255, 0, 0),
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # How mathch keypoints
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matched
    numGoodMatches = int(len(matches) * 0.1)
    matches = matches[:numGoodMatches]

    # Draw top matched
    im_matched = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches and Find Homography
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv.findHomography(points2, points1, cv.RANSAC)

    # Use homography to warp image
    height, width, channels = im1.shape
    im2_reg = cv.warpPerspective(im2, h, (width, height))

    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(im1)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(im2)
    plt.title("Before Registration")

    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(im2_reg)
    plt.title("After Registration")

    plt.show()


form_test = "./images/form_test.jpg"
scanned_test = "./images/scanned_test.jpg"
registration(form_test, scanned_test)
