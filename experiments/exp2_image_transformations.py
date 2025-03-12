import cv2
import numpy as np
import os

class ImageTransformer:
    def __init__(self, image_path: str = "image.png"):
        """
        Initialize with an image.
        
        Args:
            image_path (str): Path to the input image
        """
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.image = self.original.copy()
        
        # Create output directory
        self.output_dir = "../output/transformations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def translate(self, tx, ty):
        """
        Translate the image by tx and ty pixels.
        
        Args:
            tx (int): Translation in x direction
            ty (int): Translation in y direction
        """
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        self.image = cv2.warpAffine(self.original, matrix, 
                                   (self.original.shape[1], self.original.shape[0]))
        return self
    
    def rotate(self, angle, center=None):
        """
        Rotate the image by given angle.
        
        Args:
            angle (float): Rotation angle in degrees
            center (tuple, optional): Center of rotation
        """
        if center is None:
            center = (self.original.shape[1] // 2, self.original.shape[0] // 2)
            
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.image = cv2.warpAffine(self.original, matrix, 
                                   (self.original.shape[1], self.original.shape[0]))
        return self
    
    def scale(self, sx, sy):
        """
        Scale the image by sx and sy factors.
        
        Args:
            sx (float): Scale factor in x direction
            sy (float): Scale factor in y direction
        """
        self.image = cv2.resize(self.original, None, fx=sx, fy=sy)
        return self
    
    def affine_transform(self, pts1, pts2):
        """
        Apply affine transformation using three points.
        
        Args:
            pts1 (np.array): Source points (3x2)
            pts2 (np.array): Destination points (3x2)
        """
        matrix = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.original, matrix, 
                                   (self.original.shape[1], self.original.shape[0]))
        return self
    
    def convert_color_space(self, conversion):
        """
        Convert image to different color space.
        
        Args:
            conversion (str): One of 'hsv', 'lab', 'ycrcb'
        """
        conversion = conversion.lower()
        if conversion == 'hsv':
            self.image = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        elif conversion == 'lab':
            self.image = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        elif conversion == 'ycrcb':
            self.image = cv2.cvtColor(self.original, cv2.COLOR_BGR2YCrCb)
        else:
            raise ValueError("Unsupported color space conversion")
        return self
    
    def save(self, filename):
        """Save the current image state to the output directory"""
        output_path = os.path.join(self.output_dir, f"{filename}.jpg")
        cv2.imwrite(output_path, self.image)
        return self

def main():
    # Example usage
    try:
        transformer = ImageTransformer()
        
        # Translation
        transformer.translate(50, 50).save("translated")
        
        # Rotation
        transformer.rotate(45).save("rotated")
        
        # Scaling
        transformer.scale(1.5, 1.5).save("scaled")
        
        # Affine transformation
        rows, cols = transformer.original.shape[:2]
        pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
        pts2 = np.float32([[cols*0.2, rows*0.1], [cols*0.9, rows*0.2], [cols*0.1, rows*0.9]])
        transformer.affine_transform(pts1, pts2).save("affine")
        
        # Color space conversions
        transformer.convert_color_space('hsv').save("hsv")
        transformer.convert_color_space('lab').save("lab")
        transformer.convert_color_space('ycrcb').save("ycrcb")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 