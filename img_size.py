from PIL import Image

# Load the image
img = Image.open('/home/mmai22/ObjectClear/raw_image.png')

# Get image dimensions
width, height = img.size

print(f"Image size: {width} x {height} pixels")
print(f"Width: {width} pixels")
print(f"Height: {height} pixels")
