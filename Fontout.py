from PIL import Image, ImageDraw, ImageFont
import os

# Set the directory to save the images
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the size of the image and the background color
img_size = (100, 100)
bg_color = (255, 255, 255)

# Set the text color and size
text_color = (0, 0, 0)
text_size = 72

# Get the list of available fonts
fonts = [f for f in os.listdir('/Users/osmond/Desktop/untitled folder 4') if f.endswith('.ttf')]

# Iterate over each font
for font_name in fonts:
    # Create a font object
    font = ImageFont.truetype(f'/Library/Fonts/{font_name}', text_size)

    # Iterate over each character
    for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789¡™£¢∞§¶•ªº–≠`œ∑´®†á¨ˆø∏“‘«åíîï©óô˚ò…æ≈ç√ı˜â≤≥÷~!@#$%^&*()_+|}{":?><~':
        # Create a new image with the specified background color
        img = Image.new('RGB', img_size, bg_color)

        # Create a Draw object to draw on the image
        draw = ImageDraw.Draw(img)

        # Get the size of the text
        text_width, text_height = draw.textsize(char, font=font)

        # Calculate the position to center the text on the image
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2

        # Draw the text on the image
        draw.text((x, y), char, fill=text_color, font=font)

        # Save the image
        img.save(f'{output_dir}/{font_name}_{char}.png')
