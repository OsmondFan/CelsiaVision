import cv2
import pytesseract
from CELOCR import*
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
 
def extract_text_from_lines(lines, gray):
    # Initialize list of extracted text
    extracted_text = []

    # Extract text from each line
    for line in lines:
        # Sort text boxes in line by their x-coordinates
        line = sorted(line, key=lambda box: box[0])

        # Initialize list of words in line
        words = []

        # Extract text from each text box in line
        for box in line:
            x, y, w, h = box
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, img_size)
            data = pytesseract.image_to_string(roi_resized, lang='eng', config='--psm 6')
            words.append(data)

        # Join words in line and append to extracted text
        line_text = ' '.join(words)
        extracted_text.append(line_text)

    return extracted_text


def main():
    # Capture screenshot
    screenshot = capture_screenshot()

    # Enlarge screenshot by a factor of 4
    enlarged_screenshot = enlarge_screenshot(screenshot, 4)

    # Detect buttons and text in screenshot
    button_boxes, text_boxes = detect_buttons_text(screenshot)

    # Group text boxes into lines
    lines = group_text_boxes(text_boxes)

    # Draw bounding boxes on screenshot
    for x,y,w,h in button_boxes:
        cv2.rectangle(screenshot,(x,y),(x+w,y+h),(0,255,0),2)
    for line in lines:
        x = min([box[0] for box in line])
        y = min([box[1] for box in line])
        w = max([box[0]+box[2] for box in line]) - x
        h = max([box[1]+box[3] for box in line]) - y
        cv2.rectangle(screenshot,(x,y),(x+w,y+h),(255,0,0),2)

    # Add legend to screenshot
    screenshot_with_legend = add_legend(screenshot)

    # Display result
    cv2.imshow('Result', screenshot_with_legend)
    cv2.waitKey(0)

    # Convert screenshot to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Extract text from lines
    extracted_text = extract_text_from_lines(lines, gray)

    # Print extracted text
    for line in extracted_text:
        print(line)
