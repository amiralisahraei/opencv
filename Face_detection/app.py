import cv2 as cv
import sys
import matplotlib.pyplot as plt

# Read Pre-trained model
model = cv.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Model parameter
# The input image must be the same as the images the model has been trained on
in_width = 300
in_height = 300
mean = [104, 117, 123]
confidence_threshold = 0.5  # level of certainty or reliability


def face_detection(input_image):
    input_image_height = input_image.shape[0]
    input_image_width = input_image.shape[1]

    # Create a 4D blob from a frame.
    # Preprocessing on the input image
    blob = cv.dnn.blobFromImage(
        input_image, 1.0, (in_width, in_height), mean, swapRB=False, crop=False
    )

    # Run model on input image
    model.setInput(blob)
    detections = model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            # Multiply with width and height bcs the pixels are normalized so they are between 0 and 1
            x_left_top = int(detections[0, 0, i, 3] * input_image_width)
            y_left_top = int(detections[0, 0, i, 4] * input_image_height)
            x_right_bottom = int(detections[0, 0, i, 5] * input_image_width)
            y_right_bottom = int(detections[0, 0, i, 6] * input_image_height)

            cv.rectangle(
                input_image,
                (x_left_top, y_left_top),
                (x_right_bottom, y_right_bottom),
                (0, 255, 0),
            )
            label = f"Confidence: {confidence:.2f}"
            label_size, base_line = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv.rectangle(
                input_image,
                (x_left_top, y_left_top - label_size[1]),
                (x_left_top + label_size[0], y_left_top + base_line),
                (255, 255, 255),
                cv.FILLED,
            )
            cv.putText(
                input_image,
                label,
                (x_left_top, y_left_top),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
            )

    return input_image


# Test on a single image
def image_face_detection(test_image_path):
    test_image = cv.imread(test_image_path)
    output_image = face_detection(test_image)
    output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

    plt.imshow(output_image)
    plt.axis("off")
    plt.show()


image_face_detection("./images/couple.jpg")


# Test the model on the system camera
def camere_face_detection():
    s = 0
    if len(sys.argv) > 1:
        s = sys.argv[1]

    source = cv.VideoCapture(s)

    win_name = "Face Detection"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    while cv.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            break

        frame = cv.flip(frame, 1)
        frame = face_detection(frame)
        cv.imshow(win_name, frame)

    source.release()
    cv.destroyAllWindows()


# camere_face_detection()
