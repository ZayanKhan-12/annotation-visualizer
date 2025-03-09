import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np


# Paths
image_folder = "Datasets2/archive/images"
annotation_file = "Datasets2/archive/annotations.xml"


# Original annotation resolution (Change this if needed)
ORIGINAL_WIDTH = 1280  # Change this based on your original annotation size
ORIGINAL_HEIGHT = 720  # Change this based on your annotation size


# Load XML Annotations
tree = ET.parse(annotation_file)
root = tree.getroot()


# Define Colors for Different Labels
colors = {
   "car": (0, 0, 255),        # Red
   "road_sign": (255, 0, 0),  # Blue
   "marking": (0, 255, 255),  # Yellow
   "background": (255, 255, 255), # White
   "road_surface": (0, 255, 0)  # Green
}


# Iterate through all images in annotations
for image in root.findall("image"):
   img_name = image.get("name")
   img_path = os.path.join(image_folder, os.path.basename(img_name))  # ✅ Fixes the double "images/images/" issue


   # Check if the image exists
   if not os.path.exists(img_path):
       print(f"❌ Error: Image not found: {img_path}")
       continue


   # Load Image
   img = cv2.imread(img_path)
   if img is None:
       print(f"❌ Error: Could not load image: {img_path}")
       continue


   height, width, _ = img.shape  # Get actual image size
   print(f"📌 Displaying: {img_name} (Resolution: {width}x{height})")


   # Calculate scaling factor
   scale_x = width / ORIGINAL_WIDTH
   scale_y = height / ORIGINAL_HEIGHT


   # Draw Polygons
   for polygon in image.findall("polygon"):
       label = polygon.get("label")
       points = polygon.get("points").split(";")


       # Convert polygon points with scaling
       polygon_points = []
       for pt in points:
           x, y = map(float, pt.split(","))
           x = int(x * scale_x)  # Scale x coordinate
           y = int(y * scale_y)  # Scale y coordinate
           polygon_points.append((x, y))


       # Draw polygon
       cv2.polylines(img, [np.array(polygon_points, np.int32)], isClosed=True, color=colors.get(label, (255, 255, 255)), thickness=2)
       cv2.putText(img, label, (polygon_points[0][0], polygon_points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(label, (255, 255, 255)), 2)


   # Show Image
   cv2.imshow("Annotation Viewer", img)
   key = cv2.waitKey(0)


   # Press 'q' to quit, any other key to continue
   if key == ord('q'):
       break


cv2.destroyAllWindows()



