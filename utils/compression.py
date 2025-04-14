import cv2
import numpy as np

def compress_image_dct(img, quality=20):
    """
    Compress image using DCT (Discrete Cosine Transform)
    Lower quality means more compression
    """
    # Handle color images
    if len(img.shape) == 3:
        is_color = True
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
    else:
        is_color = False
        y = img.copy()
    
    # Apply DCT compression to the Y channel
    y = np.float32(y)
    dct = cv2.dct(y)
    
    # Zero out high-frequency components based on quality parameter
    threshold = int(min(dct.shape) * (100 - quality) / 100)
    dct[threshold:, :] = 0
    dct[:, threshold:] = 0
    
    # Inverse DCT
    idct = cv2.idct(dct)
    idct = np.uint8(np.clip(idct, 0, 255))
    
    if is_color:
        # Reconstruct the image
        compressed = cv2.merge([idct, cr, cb])
        compressed = cv2.cvtColor(compressed, cv2.COLOR_YCrCb2BGR)
    else:
        compressed = idct
    
    return compressed
