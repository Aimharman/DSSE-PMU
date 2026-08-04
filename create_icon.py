from PIL import Image, ImageDraw
import numpy as np

def create_icon():
    """Create a simple icon for the executable"""
    size = 256
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple waveform
    points = []
    for i in range(size):
        x = i
        y = size/2 + 50 * np.sin(2 * np.pi * i / 50)
        points.append((x, y))
    
    draw.line(points, fill='blue', width=5)
    draw.rectangle([50, 50, 206, 206], outline='darkblue', width=3)
    
    # Save as ICO
    img.save('pmu_icon.ico', format='ICO')
    print("Icon created: pmu_icon.ico")

if __name__ == "__main__":
    create_icon()