from datetime import datetime
import cv2
from handtrackingmodule import handTracker
import random
import json
from data_handling import run_txt_to_data, count_lines_in_file, delete_last_n_lines, store_data


def main_loop_tests():
    """
    This function will run a loop that asks the user to change their hand position and will record the data of the user doing it
    ALSO this function gets your camera and runs some deep learning to figure out the 21 points of your hand
    the points go as follows.
    0: wrist
    1-4 Thumb (from base to tip so 1 is the base of the thumb and 4 is the tip of the thumb)
    5-8 index finger
    9-12 middle finger
    13-16 ring finger
    17-20 little finger (winter is coming)
    :param: NA
    :return:
    """
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    desired_fps = 10

    frame_accumulator = []  # To store frames for the last 3 seconds
    start_time = datetime.now()  # Start time of the 5-second interval
    start_time_frames = datetime.now()
    text = ''
    font_color = (0,0,0)
    while True:

        frame_start_time = datetime.now()

        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)

        if len(lmList) != 0:
            if lmList[20][1] < 30:
                print('Left')
            elif lmList[4][1] > 600:
                print('Right')

        image = add_text_to_image(image=image, text=text, font_color=font_color)
        cv2.imshow("Video", image)

        frame_accumulator.append(lmList)

        # Calculate the time taken for processing and display
        # frame_end_time = datetime.now()
        # elapsed_time = frame_end_time - frame_start_time
        # print(f"Frame processing time: {elapsed_time}")

        # Check if 5 seconds have elapsed
        current_time = datetime.now()
        if (current_time - start_time).total_seconds() >= 5:
            start_time = current_time
            start_time_frames = current_time
            frame_accumulator = []
            rand = get_random()
            text = rand[0]
            font_color = rand[1]

        if (current_time - start_time_frames).total_seconds() >= 2:
            if not text == '':
                frames_to_send = frame_accumulator
                # print(len(frames_to_send))
                store_data(frames_to_send)
                store_data(data=text, filename='labels.txt')
                text = ''

        key = cv2.waitKey(1000 // desired_fps)

        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def add_text_to_image(image, text,font_color, position=(100, 300), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2):
    """
    Adds text to an image and returns the modified image.

    Args:
        image (numpy.ndarray): The input image.
        text (str): The text to be added.
        position (tuple): The (x, y) coordinates where the text will be placed.
        font (int): The font type (e.g., cv2.FONT_HERSHEY_SIMPLEX).
        font_scale (float): The font scale factor.
        font_color (tuple): The font color in BGR format.
        font_thickness (int): The thickness of the text.

    Returns:
        numpy.ndarray: The modified image with text.
    """

    output_image = image.copy()
    cv2.putText(output_image, text, position, font, font_scale, font_color, font_thickness)
    return output_image


def get_random():
    """
    randomly returns the text hand down, hand up, hand steady and the font color
    :return: text and font color
    """
    random_num = random.randint(0, 2)
    # 0 will be point hand down
    # 1 will be point hand up
    # 2 will be keep hand steady
    if random_num == 0:
        text = 'HAND DOWN'
        font_color = (255, 255, 0)
    elif random_num == 1:
        text = 'HAND UP'
        font_color = (255, 0, 255)
    else:
        text = 'KEEP HAND STEADY'
        font_color = (0, 255, 255)
    return text, font_color


def start_function():

    x = input('Hello, would you like to (gather) data or (view) data or (combine) data: ')
    if x.lower() == 'gather':
        print('Press q to exit out')
        delete_last_n_lines() # deletes last 3 lines of the txt file incase when shutting off you moved your hand weird
        main_loop_tests()
    if x.lower() == 'view':
        for x in range(count_lines_in_file()):
            data = run_txt_to_data(x + 1)
            print(data)
    if x.lower() == 'combine':
        data_files = ['data.txt', 'data_1.txt', 'data_2.txt', 'data_3.txt', 'data_4.txt', 'data_5.txt', 'data_6.txt',  'data_7.txt']
        total_data = []
        for data_file in data_files:
            temp_data = []
            for x in range(count_lines_in_file(filename=data_file)):
                data = run_txt_to_data(x + 1)
                temp_data.append(data)
            total_data.append(temp_data)
            # for y in total_data:
            #     for z in y:
            #         print(z)

        # creates a new file called total_data with all the data
        # for data in total_data:
        #     for data3 in data4:
        #         store_data(data3, filename='total_data.txt')

def neural_network(lst):
    print(lst)
    print()
    print()

def send_13_frames():
    """
    This function will send 13 frames worth of hand tracking data to the neural network
    :param: NA
    :return: NA
    """
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    desired_fps = 10

    frame_accumulator = []  # To store frames for the last 3 seconds
    start_time_frames = datetime.now()
    
    while True:
        # Read in image, create image with graph on hand and create list of graph points
        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)

        # Show image with graph on screen
        cv2.imshow("Video", image)
        # Add list of graph points to accumulator
        frame_accumulator.append(lmList)

        # Check if we have 13 frames
        if (len(frame_accumulator) > 13):
            frame_accumulator.pop(0)
            neural_network(frame_accumulator)

        # Check if 3 seconds have elapsed
        current_time = datetime.now()
        if (current_time - start_time_frames).total_seconds() >= 10:
            # If so, close video window and return data
            cap.release()
            cv2.destroyAllWindows()
            return

        cv2.waitKey(1000 // desired_fps)

if __name__ == '__main__':
    send_13_frames()
    # start_function()
