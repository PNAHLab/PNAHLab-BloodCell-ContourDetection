import cv2
import numpy as np

class ImageProcess:
    def convert_to_gray(self, image):
        """Convert the image to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def clahe(self, image, clahe=None):
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for enhancing contrast"""
        self.clahe = clahe if clahe is not None else cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return self.clahe.apply(image)
    
    def apply_threshold(self, image, threshold=150, method=cv2.THRESH_BINARY):
        """Apply binary thresholding to the image to create a binary (black & white) image"""
        _, binary = cv2.threshold(image, threshold, 255, method)  # Only return the binary image
        return binary
    
    def extract_contour(self, image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
        """
        Extract contours from the image. If the image is not already binary, it will be converted to grayscale 
        and thresholded before finding contours.
        """

        # Convert to grayscale and then apply threshold
        gray = self.convert_to_gray(image)
        clahe = self.clahe(image=gray)
        binary = self.apply_threshold(clahe)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, mode, method)
        return contours
    
    def filter_contours(self, min_area, contours):
        """Filter out contours that have an area smaller than the specified minimum"""
        return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    def draw_contour(self, image, contours, thickness=1):
        """Draw contours on the image"""
        copy_image = image.copy()  # Make a copy of the image to avoid modifying the original
        
        # Draw contours on the image copy
        cv2.drawContours(copy_image, contours, contourIdx=-1, color=(0, 255, 0), thickness=thickness, lineType=cv2.LINE_AA)
        
        # Display the image with contours
        self.show_image(copy_image, "Contours of your image")
    
    def get_bounding_boxes(self, contours):
        """Get bounding boxes for each contour (rectangular regions enclosing each contour)"""
        return [cv2.boundingRect(cnt) for cnt in contours]
    
    def filter_bounding_box(self, bounding_boxes, min_w, min_h):
        """
        Filter bounding boxes that have a width smaller than min_w or height smaller than min_h.
        This is useful for ignoring small objects that might not be relevant.
        """
        return [box for box in bounding_boxes if box[2] >= min_w and box[3] >= min_h]
    
    def draw_bounding_boxes(self, image, bounding_boxes):
        """Draw bounding boxes on the image"""
        image_copy = image.copy()  # Make a copy of the image
        for (x, y, w, h) in bounding_boxes:
            # Draw a rectangle for each bounding box (in blue color)
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the image with bounding boxes
        self.show_image(image_copy, 'Bounding Boxes')

    def show_image(self, image, title="Your image here."):
        """Display image"""
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()