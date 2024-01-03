import sounddevice as sd
import numpy as np

def audio_callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    if volume_norm > 0:
        print("Audio activity detected")
    else:
        print('dunno')

# Get a list of available audio input devices
devices = sd.query_devices()
input_devices = [device['name'] for device in devices if device['max_input_channels'] > 0]

# Prompt the user to choose an audio input device
print("Please choose an audio input device:")
for i, device in enumerate(input_devices):
    print(f"{i + 1}. {device}")
device_index = int(input("Enter the number of the desired device: ")) - 1
device_name = input_devices[device_index]

# Start the audio stream using the selected device
with sd.InputStream(device=device_name, callback=audio_callback):
    while True:
        pass
