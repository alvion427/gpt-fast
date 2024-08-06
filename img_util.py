import math

from PIL import Image, ImageDraw, ImageFont
import torch

def prompt_to_texture(token_strings, weights):
    # Normalize attention weights using PyTorch
    weights /= torch.max(weights)

    # Create a larger image with white background for higher resolution
    img = Image.new('RGB', (800, 400), color='white')
    d = ImageDraw.Draw(img)

    # Use a higher-quality font
    font = ImageFont.truetype("arial.ttf", 24)  # Replace 'arial.ttf' with the path to your .ttf font file

    # Starting position
    x, y = 20, 20
    line_height = 40

    # Render each word
    for word, weight in zip(token_strings, weights):
        if math.isnan(weight):
            continue

        # Compute color intensity (more attention -> brighter red)
        weight_color = min(255, max(0, int(255 * (1 - weight.item()))))
        color = (255, weight_color, weight_color)

        # Calculate text size and draw background
        text_size = d.textsize(word, font=font)
        d.rectangle([x, y, x + text_size[0], y + text_size[1]], fill=color)

        # Draw the word in black (or any other color for contrast)
        d.text((x, y), word, fill="black", font=font)

        # Update x position for next word
        x += text_size[0]

        # Line wrapping
        if x > 750:  # Adjust wrap limit based on new image size
            x = 20
            y += line_height + 10

    return img
