import cv2
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from skimage import restoration, morphology, filters, measure
from skimage.feature import canny
from sklearn.cluster import DBSCAN
import pytesseract
import warnings
warnings.filterwarnings('ignore')

class JPEGSEMProcessor:
    """
    Comprehensive processor for JPEG SEM images with text overlays and compression artifacts
    """
    
    def __init__(self):
        self.overlay_detector = OverlayDetector()
        self.jpeg_enhancer = JPEGArtifactReducer()
        self.quality_analyzer = QualityAnalyzer()
        
    def process_jpeg_sem(self, image_path, remove_overlays=True, enhance_jpeg=True):
        """
        Complete processing pipeline for JPEG SEM images
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_image = image.copy()
        
        processing_report = {
            'original_quality': {},
            'overlays_detected': [],
            'jpeg_artifacts_detected': False,
            'final_quality': {},
            'processing_steps': []
        }
        
        # Step 1: Analyze original image quality
        processing_report['original_quality'] = self.quality_analyzer.analyze_image_quality(image)
        processing_report['processing_steps'].append("Original quality analysis")
        
        # Step 2: Detect and analyze JPEG compression artifacts
        jpeg_quality = self.jpeg_enhancer.detect_jpeg_quality(image)
        processing_report['jpeg_artifacts_detected'] = jpeg_quality < 80
        processing_report['processing_steps'].append(f"JPEG quality detected: {jpeg_quality}")
        
        # Step 3: Detect text overlays and annotations
        if remove_overlays:
            overlays_info = self.overlay_detector.detect_all_overlays(image)
            processing_report['overlays_detected'] = overlays_info
            
            if overlays_info:
                print(f"Detected {len(overlays_info)} overlay regions")
                image = self.overlay_detector.remove_overlays(image, overlays_info)
                processing_report['processing_steps'].append(f"Removed {len(overlays_info)} overlay regions")
        
        # Step 4: Enhance JPEG artifacts if needed
        if enhance_jpeg and jpeg_quality < 85:
            image = self.jpeg_enhancer.reduce_artifacts(image, jpeg_quality)
            processing_report['processing_steps'].append("JPEG artifact reduction applied")
        
        # Step 5: Final quality analysis
        processing_report['final_quality'] = self.quality_analyzer.analyze_image_quality(image)
        processing_report['processing_steps'].append("Final quality analysis")
        
        return image, processing_report

class OverlayDetector:
    """
    Detects and removes text overlays, timestamps, and measurement annotations
    """
    
    def detect_all_overlays(self, image):
        """
        Comprehensive overlay detection using multiple methods
        """
        overlays = []
        
        # Method 1: Text detection using OCR
        text_regions = self._detect_text_regions(image)
        overlays.extend(text_regions)
        
        # Method 2: High contrast overlay detection
        contrast_regions = self._detect_high_contrast_overlays(image)
        overlays.extend(contrast_regions)
        
        # Method 3: Geometric pattern detection (measurement marks, scale bars)
        geometric_regions = self._detect_geometric_overlays(image)
        overlays.extend(geometric_regions)
        
        # Method 4: Corner overlay detection (timestamps, logos)
        corner_regions = self._detect_corner_overlays(image)
        overlays.extend(corner_regions)
        
        # Remove overlapping regions
        overlays = self._merge_overlapping_regions(overlays)
        
        return overlays
    
    def _detect_text_regions(self, image):
        """
        Use OCR to detect text regions
        """
        text_regions = []
        
        try:
            # Use Tesseract to detect text
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    if w > 10 and h > 10:  # Size threshold
                        text_regions.append({
                            'type': 'text',
                            'bbox': (x, y, w, h),
                            'confidence': data['conf'][i],
                            'text': data['text'][i]
                        })
        except Exception as e:
            print(f"OCR detection failed: {e}")
        
        return text_regions
    
    def _detect_high_contrast_overlays(self, image):
        """
        Detect overlays based on extreme pixel values (pure white/black text)
        """
        overlays = []
        
        # Detect very bright regions (white text)
        white_mask = image > 240
        
        # Detect very dark regions (black text on bright background)
        dark_mask = image < 15
        
        # Combine masks
        overlay_mask = white_mask | dark_mask
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        overlay_mask = cv2.morphologyEx(overlay_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        contours, _ = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter based on size and aspect ratio
            if 50 < area < 5000 and w > 10 and h > 8:
                # Check if region has text-like characteristics
                region = image[y:y+h, x:x+w]
                if self._is_text_like_region(region):
                    overlays.append({
                        'type': 'high_contrast',
                        'bbox': (x, y, w, h),
                        'confidence': 80,
                        'area': area
                    })
        
        return overlays
    
    def _detect_geometric_overlays(self, image):
        """
        Detect measurement marks, scale bars, and geometric annotations
        """
        overlays = []
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Line detection using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=5)
        
        if lines is not None:
            # Group nearby parallel lines (potential scale bars)
            line_groups = self._group_parallel_lines(lines)
            
            for group in line_groups:
                if len(group) >= 2:  # At least 2 parallel lines
                    # Calculate bounding box for the group
                    all_points = np.concatenate(group)
                    x_min, y_min = np.min(all_points, axis=0)
                    x_max, y_max = np.max(all_points, axis=0)
                    
                    # Check if this looks like a scale bar
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    if width > 50 and height < 30:  # Horizontal scale bar
                        overlays.append({
                            'type': 'scale_bar',
                            'bbox': (x_min-5, y_min-5, width+10, height+10),
                            'confidence': 70,
                            'orientation': 'horizontal'
                        })
                    elif height > 50 and width < 30:  # Vertical scale bar
                        overlays.append({
                            'type': 'scale_bar',
                            'bbox': (x_min-5, y_min-5, width+10, height+10),
                            'confidence': 70,
                            'orientation': 'vertical'
                        })
        
        return overlays
    
    def _detect_corner_overlays(self, image):
        """
        Detect overlays typically found in corners (timestamps, logos, instrument info)
        """
        overlays = []
        h, w = image.shape
        
        # Define corner regions (10% of image from each corner)
        corner_size_h = int(h * 0.15)
        corner_size_w = int(w * 0.15)
        
        corners = [
            (0, 0, corner_size_w, corner_size_h),  # Top-left
            (w-corner_size_w, 0, corner_size_w, corner_size_h),  # Top-right
            (0, h-corner_size_h, corner_size_w, corner_size_h),  # Bottom-left
            (w-corner_size_w, h-corner_size_h, corner_size_w, corner_size_h)  # Bottom-right
        ]
        
        for i, (x, y, cw, ch) in enumerate(corners):
            corner_region = image[y:y+ch, x:x+cw]
            
            # Check for text-like patterns in corners
            if self._has_corner_overlay(corner_region):
                overlays.append({
                    'type': 'corner_overlay',
                    'bbox': (x, y, cw, ch),
                    'confidence': 60,
                    'corner': i
                })
        
        return overlays
    
    def _is_text_like_region(self, region):
        """
        Check if a region has text-like characteristics
        """
        if region.size == 0:
            return False
        
        # Calculate edge density
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate contrast
        contrast = region.std()
        
        # Text typically has high edge density and contrast
        return edge_density > 0.02 and contrast > 30
    
    def _has_corner_overlay(self, corner_region):
        """
        Check if corner region contains overlay information
        """
        # Check for uniform regions (common in overlays)
        hist = cv2.calcHist([corner_region], [0], None, [256], [0, 256])
        
        # Check for bimodal distribution (text on background)
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 50:
                peaks.append(i)
        
        # Text regions often have 2 main peaks (text and background)
        return len(peaks) >= 2
    
    def _group_parallel_lines(self, lines):
        """
        Group parallel lines that might form scale bars or measurement marks
        """
        if lines is None:
            return []
        
        # Calculate line parameters
        line_params = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            line_params.append((angle, length, center, line[0]))
        
        # Group lines with similar angles
        angle_threshold = 0.2  # radians
        groups = []
        used = set()
        
        for i, (angle1, length1, center1, line1) in enumerate(line_params):
            if i in used:
                continue
            
            group = [line1]
            used.add(i)
            
            for j, (angle2, length2, center2, line2) in enumerate(line_params):
                if j in used:
                    continue
                
                if abs(angle1 - angle2) < angle_threshold:
                    group.append(line2)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _merge_overlapping_regions(self, overlays):
        """
        Merge overlapping overlay regions
        """
        if not overlays:
            return []
        
        # Sort by area (larger first)
        overlays.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        
        merged = []
        for overlay in overlays:
            x, y, w, h = overlay['bbox']
            overlapped = False
            
            for merged_overlay in merged:
                mx, my, mw, mh = merged_overlay['bbox']
                
                # Check for overlap
                if (x < mx + mw and x + w > mx and 
                    y < my + mh and y + h > my):
                    
                    # Calculate intersection over union
                    intersection = max(0, min(x + w, mx + mw) - max(x, mx)) * \
                                  max(0, min(y + h, my + mh) - max(y, my))
                    union = w * h + mw * mh - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.3:  # Significant overlap
                        # Merge bounding boxes
                        new_x = min(x, mx)
                        new_y = min(y, my)
                        new_w = max(x + w, mx + mw) - new_x
                        new_h = max(y + h, my + mh) - new_y
                        
                        merged_overlay['bbox'] = (new_x, new_y, new_w, new_h)
                        merged_overlay['confidence'] = max(overlay['confidence'], 
                                                         merged_overlay['confidence'])
                        overlapped = True
                        break
            
            if not overlapped:
                merged.append(overlay)
        
        return merged
    
    def remove_overlays(self, image, overlays):
        """
        Remove detected overlays using inpainting
        """
        if not overlays:
            return image
        
        # Create mask for all overlay regions
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for overlay in overlays:
            x, y, w, h = overlay['bbox']
            # Add some padding to ensure complete removal
            padding = 3
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            mask[y:y+h, x:x+w] = 255
        
        # Apply morphological operations to smooth mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Use inpainting to remove overlays
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return result

class JPEGArtifactReducer:
    """
    Reduces JPEG compression artifacts using various techniques
    """
    
    def detect_jpeg_quality(self, image):
        """
        Estimate JPEG quality based on blocking artifacts
        """
        # Analyze 8x8 block patterns typical in JPEG
        h, w = image.shape
        block_size = 8
        
        blocking_score = 0
        block_count = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = image[y:y+block_size, x:x+block_size].astype(np.float32)
                
                # Check for blocking artifacts at block boundaries
                if x + block_size < w:
                    # Horizontal boundary
                    left_edge = block[:, -1]
                    right_block = image[y:y+block_size, x+block_size:x+2*block_size]
                    if right_block.shape[1] > 0:
                        right_edge = right_block[:, 0]
                        h_diff = np.mean(np.abs(left_edge - right_edge))
                        blocking_score += h_diff
                        block_count += 1
                
                if y + block_size < h:
                    # Vertical boundary
                    top_edge = block[-1, :]
                    bottom_block = image[y+block_size:y+2*block_size, x:x+block_size]
                    if bottom_block.shape[0] > 0:
                        bottom_edge = bottom_block[0, :]
                        v_diff = np.mean(np.abs(top_edge - bottom_edge))
                        blocking_score += v_diff
                        block_count += 1
        
        avg_blocking = blocking_score / block_count if block_count > 0 else 0
        
        # Convert blocking score to estimated quality (empirical mapping)
        if avg_blocking < 2:
            quality = 95
        elif avg_blocking < 5:
            quality = 85
        elif avg_blocking < 10:
            quality = 70
        elif avg_blocking < 20:
            quality = 50
        else:
            quality = 30
        
        return quality
    
    def reduce_artifacts(self, image, estimated_quality):
        """
        Reduce JPEG artifacts based on estimated quality
        """
        if estimated_quality > 85:
            return image  # High quality, no need for enhancement
        
        result = image.copy().astype(np.float32)
        
        # Method 1: Bilateral filtering to reduce blocking while preserving edges
        if estimated_quality < 70:
            result = cv2.bilateralFilter(result.astype(np.uint8), d=5, 
                                       sigmaColor=30, sigmaSpace=30).astype(np.float32)
        
        # Method 2: Wavelet denoising for moderate artifacts
        if estimated_quality < 60:
            from skimage.restoration import denoise_wavelet
            result = denoise_wavelet(result/255.0, sigma=0.1, 
                                   wavelet='db4', mode='soft') * 255
        
        # Method 3: Edge-preserving smoothing for severe artifacts
        if estimated_quality < 40:
            # Use edge-preserving filter
            result = cv2.edgePreservingFilter(result.astype(np.uint8), 
                                            flags=2, sigma_s=50, sigma_r=0.4)
        
        return result.astype(np.uint8)

class QualityAnalyzer:
    """
    Analyzes image quality metrics considering JPEG-specific issues
    """
    
    def analyze_image_quality(self, image):
        """
        Comprehensive quality analysis for JPEG SEM images
        """
        metrics = {}
        
        # Basic quality metrics
        metrics['contrast'] = self._calculate_contrast(image)
        metrics['sharpness'] = self._calculate_sharpness(image)
        metrics['noise_level'] = self._estimate_noise(image)
        
        # JPEG-specific metrics
        metrics['blocking_artifacts'] = self._detect_blocking_artifacts(image)
        metrics['ringing_artifacts'] = self._detect_ringing_artifacts(image)
        
        # Overall quality score
        metrics['quality_score'] = self._calculate_overall_quality(metrics)
        
        return metrics
    
    def _calculate_contrast(self, image):
        """RMS contrast calculation"""
        return float(np.std(image))
    
    def _calculate_sharpness(self, image):
        """Laplacian variance for sharpness"""
        laplacian = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
        return float(laplacian.var())
    
    def _estimate_noise(self, image):
        """Estimate noise using local standard deviation"""
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        noise_std = np.std(image - local_mean)
        return float(noise_std)
    
    def _detect_blocking_artifacts(self, image):
        """Detect JPEG blocking artifacts"""
        # Similar to quality estimation but return as metric
        h, w = image.shape
        block_size = 8
        
        blocking_score = 0
        block_count = 0
        
        for y in range(block_size, h - block_size, block_size):
            for x in range(block_size, w - block_size, block_size):
                # Check discontinuities at block boundaries
                h_diff = np.mean(np.abs(image[y-1:y+1, x-block_size:x+block_size].astype(np.float32).diff(axis=0)))
                v_diff = np.mean(np.abs(image[y-block_size:y+block_size, x-1:x+1].astype(np.float32).diff(axis=1)))
                
                blocking_score += (h_diff + v_diff) / 2
                block_count += 1
        
        return blocking_score / block_count if block_count > 0 else 0
    
    def _detect_ringing_artifacts(self, image):
        """Detect ringing artifacts around edges"""
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Dilate edges to create search regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_regions = cv2.dilate(edges, kernel)
        
        # Calculate variance in edge regions (ringing creates oscillations)
        edge_pixels = image[edge_regions > 0]
        if len(edge_pixels) > 0:
            return float(np.var(edge_pixels))
        return 0
    
    def _calculate_overall_quality(self, metrics):
        """Calculate overall quality score (0-100)"""
        # Normalize metrics and combine
        contrast_score = min(100, metrics['contrast'] / 50 * 100)
        sharpness_score = min(100, metrics['sharpness'] / 1000 * 100)
        noise_penalty = max(0, 100 - metrics['noise_level'] * 2)
        blocking_penalty = max(0, 100 - metrics['blocking_artifacts'] * 10)
        ringing_penalty = max(0, 100 - metrics['ringing_artifacts'] / 100)
        
        overall = (contrast_score + sharpness_score + noise_penalty + 
                  blocking_penalty + ringing_penalty) / 5
        
        return min(100, max(0, overall))

# Example usage and testing
def demonstrate_jpeg_processing():
    """
    Demonstrate the JPEG SEM processing pipeline
    """
    # Initialize processor
    processor = JPEGSEMProcessor()
    
    # Create a synthetic test image with overlays
    test_image = create_synthetic_sem_with_overlays()
    
    # Save test image as JPEG with compression
    cv2.imwrite('test_sem_with_overlays.jpg', test_image, 
                [cv2.IMWRITE_JPEG_QUALITY, 60])
    
    # Process the JPEG image
    processed_image, report = processor.process_jpeg_sem(
        'test_sem_with_overlays.jpg',
        remove_overlays=True,
        enhance_jpeg=True
    )
    
    # Display results
    print("Processing Report:")
    print(f"Original quality score: {report['original_quality']['quality_score']:.1f}")
    print(f"Overlays detected: {len(report['overlays_detected'])}")
    print(f"JPEG artifacts detected: {report['jpeg_artifacts_detected']}")
    print(f"Final quality score: {report['final_quality']['quality_score']:.1f}")
    print(f"Processing steps: {report['processing_steps']}")
    
    return processed_image, report

def create_synthetic_sem_with_overlays():
    """
    Create a synthetic SEM image with typical overlays for testing
    """
    # Create base SEM-like image
    base_image = np.random.randint(50, 200, (1024, 1024), dtype=np.uint8)
    
    # Add some structures
    cv2.circle(base_image, (300, 300), 50, 220, -1)
    cv2.rectangle(base_image, (500, 500), (600, 550), 180, -1)
    
    # Add text overlays (simulate timestamp)
    cv2.putText(base_image, "2024-07-02 14:30:25", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    # Add measurement scale
    cv2.line(base_image, (50, 950), (150, 950), 255, 3)
    cv2.putText(base_image, "100nm", (60, 940), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    
    # Add measurement annotation
    cv2.putText(base_image, "CD: 45.2nm", (700, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    
    return base_image

if __name__ == "__main__":
    # Run demonstration
    processed, report = demonstrate_jpeg_processing()
    
    print("\nJPEG SEM Processing demonstration completed!")
    print("Check the output files for results.")
