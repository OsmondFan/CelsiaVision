import cv2
import numpy as np
from sklearn import svm
import os
import pyautogui
from concurrent.futures import ThreadPoolExecutor
import random
from joblib import dump,load
import threading
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def load_data():
    # Set the desired image shape
    img_shape = (100, 100)

    # Get list of all directories in current directory
    dir_names = [d for d in os.listdir() if os.path.isdir(d) and '_negative' not in d and 'window' not in d and 'scripts' not in d and 'cv' in d]

    # Initialize lists to store image data and labels
    X = []
    y = []

    def load_and_resize_image(f):
        return cv2.resize(cv2.imread(os.path.join(dir_name, f), 0), img_shape)

    # Iterate over directories
    for i, dir_name in enumerate(dir_names):
        # Load and resize training data
        with ThreadPoolExecutor(max_workers=20) as executor:
            images = list(executor.map(load_and_resize_image, [f for f in os.listdir(dir_name) if f.endswith('.jpg') or f.endswith('.png')]))

        # Check if any images were loaded
        if not images:
            print(f'Error: No images found in {dir_name}')
            continue

        # Flatten images and create labels
        X.extend([img.flatten() for img in images])
        y.extend([i]*len(images))

    # Convert lists to arrays
    X = np.array(X)
    y = np.array(y)

    return X, y, dir_names

def train_classifier(X, y):
    # Train SVM classifier
    clf = svm.SVC()
    clf.fit(X, y)

    # Save positive SVM classifier
    dump(clf, 'countour.joblib')


def process_screenshot(clf, dir_names, min_size, max_size):
    # Set the desired image shape
    img_shape = (100, 100)

    # Take a screenshot of the current desktop
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Convert screenshot to grayscale
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Find contours in the image
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize list to store average confidence scores for each class
    avg_scores = np.zeros(len(dir_names))

    # Initialize counter for number of ROIs processed
    num_rois = 0

    # Initialize list to store drawn boxes
    drawn_boxes = []
    details = []
    def draw_row_box(boxes):
        # Initialize list to store drawn boxes
        drawn_boxes = []
        
        # Initialize list to store y-coordinates of boxes
        y_coords = []

        # Iterate over boxes
        for box in boxes:
            # Add y-coordinate of box to list
            y_coords.append(box['y'])

        # Compute histogram of y-coordinates
        hist, bin_edges = np.histogram(y_coords, bins=range(0, screenshot.shape[0], 10))

        # Find indices of bins with non-zero counts
        non_zero_bins = np.nonzero(hist)[0]

        # Find densest region of boxes
        max_count = 0
        max_region_start = 0
        max_region_end = 0
        for i in range(len(non_zero_bins) - 1):
            count = sum(hist[non_zero_bins[i]:non_zero_bins[i+1]+1])
            if count > max_count:
                max_count = count
                max_region_start = bin_edges[non_zero_bins[i]]
                max_region_end = bin_edges[non_zero_bins[i+1]+1]

        # Draw box around densest region on screenshot
        cv2.rectangle(screenshot, (0, max_region_start), (screenshot.shape[1], max_region_end), (255, 0, 0), 2)
        
        # Perform OCR on the boxed area
        boxed_area = screenshot[max_region_start:max_region_end, 0:screenshot.shape[1]]
        ocr_content = pytesseract.image_to_string(boxed_area)
        
        # Store box coordinates and OCR content in list
        drawn_boxes.append((0, max_region_start, screenshot.shape[1], max_region_end, ocr_content))
        
        return drawn_boxes
        

    def draw_circle(contour):
        # Compute bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Convert floating-point values to integers
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Check if contour is within specified size range
        if min_size <= w*h <= max_size:
            # Compute center and radius of circle that encloses the contour
            (c_x,c_y), radius = cv2.minEnclosingCircle(contour)
            center = (int(c_x),int(c_y))
            radius = int(radius)
            
            # Extract ROI from grayscale image
            roi = gray[y:y+h, x:x+w]
            roi_flat = cv2.resize(roi, img_shape).flatten().reshape(1, -1)
            
            # Predict type of contour using trained classifier
            scores = clf.decision_function(roi_flat)[0]
            max_score_index = np.argmax(scores)
            
            # Generate color based on directory name using hash function
            color_str = dir_names[max_score_index]
            color_hash = hash(color_str) & 0xffffff
            color_bgr = ((color_hash >> 16) & 0xff), ((color_hash >> 8) & 0xff), (color_hash & 0xff)
            
            # Draw circle around contour using generated color
            cv2.circle(screenshot, center, radius, color_bgr, 2)

            # Save image of portion circled to corresponding contour folder
            contour_dir_name = f"{dir_names[max_score_index]}"
            if not os.path.exists(contour_dir_name):
                os.makedirs(contour_dir_name)
            cv2.imwrite(os.path.join(contour_dir_name, str(random.randint(1,999999999))+'.png'), roi)
    
    def process_contour(contour):
        nonlocal avg_scores, num_rois, drawn_boxes
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if ROI is within specified size range
        if min_size <= w*h <= max_size:
            roi = gray[y:y+h, x:x+w]
            roi_flat = cv2.resize(roi, img_shape).flatten().reshape(1, -1)
            scores = clf.decision_function(roi_flat)[0]
            
            # Update average confidence scores for each class
            avg_scores += scores
            
            # Increment counter for number of ROIs processed
            num_rois += 1
            
            max_score_index = np.argmax(scores)
            
            # Generate color based on directory name using hash function
            color_str = dir_names[max_score_index]
            color_hash = hash(color_str) & 0xffffff
            color_bgr = ((color_hash >> 16) & 0xff), ((color_hash >> 8) & 0xff), (color_hash & 0xff)
            
            # Draw rectangle using generated color
            cv2.rectangle(screenshot, (x,y), (x+w,y+h), color_bgr ,2)
            
            # Add new box to list of drawn boxes
            drawn_boxes.append({'x': x, 'y': y, 'w': w, 'h': h})
            
            # Save image of portion circled to corresponding negative contour folder
            negative_dir_name = f"{dir_names[max_score_index]}_negative"
            if not os.path.exists(negative_dir_name):
                os.makedirs(negative_dir_name)
            cv2.imwrite(os.path.join(negative_dir_name, str(random.randint(1,999999999))+'.png'), roi)

    # Process contours in parallel using ThreadPoolExecutor with 20 threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_contour, contours)

    # Draw circles around all contours
    for contour in contours:
        draw_circle(contour)

    # Compute average confidence scores for each class
    if num_rois > 0:
        avg_scores /= num_rois
    else:
        print('Warning: No ROIs were processed')

    # Draw row box on screenshot
    print(drawn_boxes)
    draw_row_box(drawn_boxes)

    return screenshot, avg_scores, details


def draw_menu(screenshot, dir_names, menu_height, menu_width):
    # Create blank menu image
    menu_img = np.zeros((menu_height, menu_width, 3), dtype=np.uint8)

    for i in range(len(dir_names)):
        # Generate color based on directory name using hash function
        color_str = dir_names[i]
        color_hash = hash(color_str) & 0xffffff
        color_bgr = ((color_hash >> 16) & 0xff), ((color_hash >> 8) & 0xff), (color_hash & 0xff)
        
        # Draw color swatch and directory name on menu image
        cv2.rectangle(menu_img, (10,i*20+10), (30,i*20+30), color_bgr ,-1)
        cv2.putText(menu_img, dir_names[i], (40,i*20+25), cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255,255,255),1,cv2.LINE_AA)

    # Add menu to displayed image
    screenshot[-menu_height-10:-10,-menu_width-10:-10] = menu_img

    return screenshot

def display_screenshot(screenshot):
    # Show the resulting image
    cv2.imshow('Screenshot', screenshot)
    
    # Wait for key press
    key = cv2.waitKey(0)

    return key



# Load data and train classifiers
X, y, dir_names = load_data()
train_classifier(X, y)
clf = load('countour.joblib')

# Set menu dimensions
menu_height = len(dir_names)*20 + 20
menu_width = 200

while True:
    # Process screenshot
    min_size = int(input('Enter minimum size for ROIs (width*height): '))
    max_size = int(input('Enter maximum size for ROIs (width*height): '))
    screenshot, avg_scores, details = process_screenshot(clf, dir_names, min_size, max_size)

    print(details)
    # Draw menu on screenshot
    screenshot = draw_menu(screenshot, dir_names, menu_height, menu_width)

    # Display screenshot and get key press from user
    key = display_screenshot(screenshot)
    
    # Check if pressed key is 'r'
    if key == ord('r'):
        # Run the code again
        continue
    else:
        # Exit the loop
        break

        break
