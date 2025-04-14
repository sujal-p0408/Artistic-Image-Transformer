import cv2
import numpy as np
from sklearn.cluster import KMeans

ASCII_CHARS = ["@", "#", "%", "?", "*", "+", ";", ":", ",", "."]

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def pop_art_style(img, n_colors=5, background_color="orange"):
    """
    Create pop art style image that matches the reference images with halftone pattern
    """
    # Resize image for consistent processing
    img = cv2.resize(img, (512, 512))
    
    # Convert to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply bilateral filter to smooth while preserving edges
    img_filtered = cv2.bilateralFilter(img_rgb, 9, 75, 75)
    
    # Reshape the image for k-means clustering
    pixels = img_filtered.reshape((-1, 3))
    
    # Perform k-means clustering to reduce colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the colors and replace each pixel with its closest color
    colors = kmeans.cluster_centers_
    labels = kmeans.predict(pixels)
    quantized = colors[labels].reshape(img_rgb.shape).astype(np.uint8)
    
    # Create background color map
    bg_colors = {
        "orange": [255, 165, 0], 
        "red": [255, 0, 0], 
        "blue": [0, 0, 255],
        "yellow": [255, 255, 0], 
        "green": [0, 255, 0], 
        "purple": [128, 0, 128]
    }
    bg_color = np.array(bg_colors.get(background_color.lower(), [255, 165, 0]))
    
    # Create a background of the selected color
    background = np.ones_like(quantized) * bg_color
    background = background.astype(np.uint8)  # Ensure uint8 type
    
    # Apply halftone dot pattern (essential for pop art look)
    gray = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2GRAY)
    dots = np.zeros_like(gray)
    
    for i in range(0, gray.shape[0], 6):
        for j in range(0, gray.shape[1], 6):
            if i+6 <= gray.shape[0] and j+6 <= gray.shape[1]:
                block = gray[i:i+6, j:j+6]
                avg_val = np.mean(block)
                radius = int(5 * (255 - avg_val) / 255)
                if radius > 0:
                    cv2.circle(dots, (j+3, i+3), radius, 255, -1)
    
    # Use dots as a mask to combine quantized image and background
    dots_rgb = cv2.cvtColor(dots, cv2.COLOR_GRAY2RGB)
    mask = dots_rgb > 0
    
    # Create result array with proper type
    result = background.copy()
    np.copyto(result, quantized, where=mask)
    
    # Edge detection for bold outlines
    edges = cv2.Canny(cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY), 100, 200)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Add black edges on top
    result[edges != 0] = [0, 0, 0]
    
    # Enhance contrast and color saturation (avoiding LAB conversion)
    result = np.clip(result.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def bit_plane_slicing(img):
    """
    Perform bit plane slicing using OpenCV
    """
    planes = []
    for i in range(8):
        plane = np.uint8(np.bitwise_and(img, 2**i))
        planes.append(plane)
    
    combined = np.zeros_like(img)
    for i in [5, 6, 7]:
        combined = cv2.bitwise_or(combined, planes[i])
    return combined



def inksplash_style(img):
    """
    Create ink splash style image with improved edge definition and contrast
    """
    # Convert to grayscale for base effect
    gray = grayscale(img)
    
    # Apply bilateral filter with adjusted parameters for better edge preservation
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Enhance contrast with normalization
    stretched = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply bit plane slicing
    sliced = bit_plane_slicing(stretched)
    
    # Improved edge detection for ink effect
    edges = cv2.Sobel(stretched, cv2.CV_8U, 1, 1, ksize=3)
    edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    # Combine effects with better contrast
    result = cv2.subtract(sliced, edges)
    
    return result



def sketch_style(img, intensity=0.8):
    """
    Create pencil sketch style image
    """
    # Convert to grayscale
    gray = grayscale(img)
    
    # Invert grayscale image
    inverted = 255 - gray
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred = 255 - blurred
    
    # Create pencil sketch by dividing grayscale by inverted blurred image
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    # Enhance contrast based on intensity parameter
    alpha = intensity  # Contrast control
    beta = 10  # Brightness control
    sketch = cv2.convertScaleAbs(sketch, alpha=alpha, beta=beta)
    
    return sketch

def ascii_art(img, width=100, height=50):
    """
    Convert image to ASCII art with improved character mapping
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to make output readable
    resized = cv2.resize(gray, (width, height))
    
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(resized)
    
    # Map pixel values to ASCII characters
    ascii_str = ""
    for row in equalized:
        line = ""
        for pixel in row:
            # Map pixel value (0-255) to ASCII character index
            index = min(len(ASCII_CHARS) - 1, pixel * len(ASCII_CHARS) // 256)
            line += ASCII_CHARS[index] + " "  # Add space for better proportions
        ascii_str += line + "\n"
    
    return ascii_str
