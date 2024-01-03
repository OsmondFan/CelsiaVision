import pyautogui
import os
import subprocess,time


def move_mouse(x, y):
    pyautogui.moveTo(x, y)

def click_mouse(x, y):
    pyautogui.click(x, y)

def move_mouse_relative(x_offset, y_offset):
    pyautogui.moveRel(x_offset, y_offset)

def drag_mouse(x, y):
    pyautogui.dragTo(x, y)

def drag_mouse_relative(x_offset, y_offset):
    pyautogui.dragRel(x_offset, y_offset)

def press_key(key):
    pyautogui.press(key)

def press_key_combination(*keys):
    pyautogui.hotkey(*keys)

def type_text(text, interval=0):
    pyautogui.typewrite(text, interval=interval)

def press_and_hold_key_combination(*keys):
    for key in keys:
        pyautogui.keyDown(key)

def release_key_combination(*keys):
    for key in keys:
        pyautogui.keyUp(key)

def scroll_mouse(amount):
    pyautogui.scroll(amount)
    
'''background'''
# Define a function that runs an AppleScript command and returns the output
def run_applescript(command):
    # Use osascript to run the command
    process = subprocess.run(["osascript", "-e", command], capture_output=True)
    # Decode and strip the output
    output = process.stdout.decode().strip()
    # Return the output
    return output

#To go to Edge browser to ask bing
# Get a list of all window names
window_names = run_applescript('tell application "System Events" to get name of every window of every process')


'''N Script Area'''
def N(commands):
    for command in commands:
        if command.startswith("move_mouse"):
            x, y = map(int, command.split()[1:])
            move_mouse(x, y)
        elif command.startswith("click_mouse"):
            x, y = map(int, command.split()[1:])
            click_mouse(x, y)
        elif command.startswith("move_mouse_relative"):
            x_offset, y_offset = map(int, command.split()[1:])
            move_mouse_relative(x_offset, y_offset)
        elif command.startswith("drag_mouse"):
            x, y = map(int, command.split()[1:])
            drag_mouse(x, y)
        elif command.startswith("drag_mouse_relative"):
            x_offset, y_offset = map(int, command.split()[1:])
            drag_mouse_relative(x_offset, y_offset)
        elif command.startswith("press_key"):
            key = command.split()[1]
            press_key(key)
        elif command.startswith("press_key_combination"):
            keys = command.split()[1:]
            press_key_combination(*keys)
        elif command.startswith("type_text"):
            text = ' '.join(command.split()[1:])
            type_text(text)
        elif command.startswith("press_and_hold_key_combination"):
            keys = command.split()[1:]
            for key in keys:
                pyautogui.keyDown(key)
            for key in keys:
                pyautogui.keyUp(key)
        elif command.startswith("wait"):
            keys = command.split()[1:]
            time.sleep(int(keys[0]))
        elif command.startswith("scroll_mouse"):
            amount = int(command.split()[1])
            scroll_mouse(amount)


open_bing = [
    "press_and_hold_key_combination command space",
    "release_key_combination command space",
    "type_text edge",
    "press_key enter",
    "press_key enter",
    "type_text www.bing.com ",
    "press_key enter",
    "wait 2",
    "type_text Hi",
    "press_key enter",
    "wait 2",
    "scroll_mouse 2000"
]
open_chatgpt = [
    "press_and_hold_key_combination command space",
    "release_key_combination command space",
    "type_text safari",
    "press_key enter",
    "press_key enter",
    "type_text chat.openai.com ",
    "press_key enter",
]
while True:
    command = input().split(',')
    N(command)
    
