"""
Create test digit images to upload to the app
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Create output directory
os.makedirs('test_images', exist_ok=True)

# Create images for digits 0-9
# White digits on black background (matches MNIST format)
for digit in range(10):
    # Create 280x280 image (same as canvas size)
    img = Image.new('L', (280, 280), color=0)  # Black background
    draw = ImageDraw.Draw(img)

    # Draw the digit in white
    # Use a large font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 200)
    except:
        font = ImageFont.load_default()

    # Draw centered
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (280 - text_width) // 2 - bbox[0]
    y = (280 - text_height) // 2 - bbox[1]

    draw.text((x, y), text, fill=255, font=font)

    # Save
    img.save(f'test_images/digit_{digit}.png')
    print(f"Created test_images/digit_{digit}.png")

print("\nTest images created! Upload these to the Streamlit app.")
print("They are white digits on black background, matching MNIST format.")
