"""
Complete CD-SEM Image Quality Analyzer
Supports both TIFF (raw) and JPEG (with overlay removal) formats
Optimized for CPU-only processing with comprehensive quality metrics
"""

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import restoration, morphology, filters, measure
from skimage.feature import canny
from skimage.restoration import denoise_wavelet
from sklearn.cluster import DBSCAN
import pytesseract
import os
import time
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

class CDSEMImageAnalyzer:
    """
    Complete CD-SEM Image Quality Analyzer supporting both TIFF and JPEG formats
    """
    
    def __init__(self, use_visualization=False, debug_dir="debug_output"):
        self.tiff_processor = TIFFProcessor()
        self.jpeg_processor = JPEGProcessor(use_visualization, debug_dir)
        self.quality_analyzer = QualityAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.drift_compensator = DriftCompensation()
        
    def analyze_image(self, image_path, reference_image_path=None, remove_overlays=True):
        """
        Main analysis method that automatically detects format and processes accordingly
        """
        # Detect image format
        file_ext = os.path.splitext(image_path)[1].lower()
        
        start_time = time.time()
        
        if file_ext in ['.tif', '.tiff']:
            result = self._analyze_tiff(image_path, reference_image_path)
        elif file_ext in ['.jpg', '.jpeg']:
            result = self._analyze_jpeg(image_path, reference_image_path, remove_overlays)
        else:
            raise ValueError(f"Unsupported image format: {file_ext}")
        
        # Add processing time
        result['processing_time_ms'] = (time.time() - start_time) * 1000
        result['image_format'] = file_ext
        
        return result
    
    def _analyze_tiff(self, image_path, reference_image_path=None):
        """
        Analyze TIFF image (raw SEM data)
        """
        # Load TIFF with metadata
        image, pixel_size, metadata = self.tiff_processor.load_sem_tiff(image_path)
        
        # Basic quality metrics
        quality_metrics = self.quality_analyzer.calculate_comprehensive_metrics(image)
        
        # Pattern-specific analysis
        pattern_results = self.pattern_analyzer.analyze_patterns(image, pixel_size)
        
        # Drift compensation if reference provided
        drift_info = {}
        if reference_image_path:
            ref_image, _, _ = self.tiff_processor.load_sem_tiff(reference_image_path)
            drift_info = self.drift_compensator.analyze_drift(image, ref_image)
            # Compensate quality metrics for drift
            quality_metrics = self.drift_compensator.compensate_quality_metrics(
                quality_metrics, drift_info.get('drift_magnitude', 0))
        
        return {
            'image_path': image_path,
            'quality_metrics': quality_metrics,
            'pattern_analysis': pattern_results,
            'drift_analysis': drift_info,
            'metadata': metadata,
            'pixel_size_nm': pixel_size,
            'overlays_removed': None,  # Not applicable for TIFF
            'analysis_area_percentage': 100.0  # Full image analyzed
        }
    
    def _analyze_jpeg(self, image_path, reference_image_path=None, remove_overlays=True):
        """
        Analyze JPEG image (with potential overlays)
        """
        # Process JPEG with overlay removal
        if remove_overlays:
            processed_image, overlays, jpeg_quality = self.jpeg_processor.process_jpeg_sem(image_path)
        else:
            processed_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            overlays = []
            jpeg_quality = self.jpeg_processor.artifact_reducer.detect_jpeg_quality(processed_image)
        
        # Estimate pixel size (not available in JPEG metadata typically)
        pixel_size = 1.0  # Default, should be calibrated for your system
        
        # Quality analysis on cleaned image
        quality_metrics = self.quality_analyzer.calculate_comprehensive_metrics(processed_image)
        
        # Pattern analysis
        pattern_results = self.pattern_analyzer.analyze_patterns(processed_image, pixel_size)
        
        # Drift analysis if reference provided
        drift_info = {}
        if reference_image_path:
            if remove_overlays:
                ref_processed, _, _ = self.jpeg_processor.process_jpeg_sem(reference_image_path)
            else:
                ref_processed = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
            
            drift_info = self.drift_compensator.analyze_drift(processed_image, ref_processed)
            quality_metrics = self.drift_compensator.compensate_quality_metrics(
                quality_metrics, drift_info.get('drift_magnitude', 0))
        
        # Calculate analysis area
        if overlays:
            total_overlay_area = sum(o['bbox'][2] * o['bbox'][3] for o in overlays)
            image_area = processed_image.shape[0] * processed_image.shape[1]
            analysis_area_percentage = ((image_area - total_overlay_area) / image_area) * 100
        else:
            analysis_area_percentage = 100.0
        
        return {
            'image_path': image_path,
            'quality_metrics': quality_metrics,
            'pattern_analysis': pattern_results,
            'drift_analysis': drift_info,
            'metadata': {'jpeg_quality': jpeg_quality},
            'pixel_size_nm': pixel_size,
            'overlays_removed': overlays,
            'analysis_area_percentage': analysis_area_percentage
        }
    
    def batch_analyze(self, image_paths, reference_path=None, max_workers=None):
        """
        Analyze multiple images in parallel
        """
        if max_workers is None:
            max_workers = min(cpu_count(), 4)
        
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.analyze_image, path, reference_path): path 
                for path in image_paths
            }
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ Analyzed: {os.path.basename(path)}")
                except Exception as e:
                    print(f"❌ Failed to analyze {path}: {e}")
                    results.append({
                        'image_path': path,
                        'error': str(e),
                        'success': False
                    })
        
        return results

class TIFFProcessor:
    """
    Handles TIFF image loading and metadata extraction
    """
    
    def load_sem_tiff(self, filepath):
        """
        Load SEM TIFF image with metadata preservation
        """
        try:
            with tifffile.TiffFile(filepath) as tif:
                image = tif.asarray()
                metadata = self._extract_metadata(tif)
                
                # Extract pixel size from metadata
                pixel_size = metadata.get('pixel_size_nm', 1.0)
                
                # Ensure grayscale
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                return image, pixel_size, metadata
                
        except Exception as e:
            print(f"Error loading TIFF: {e}")
            # Fallback to OpenCV
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {filepath}")
            
            return image, 1.0, {}
    
    def _extract_metadata(self, tif):
        """
        Extract vendor-specific metadata from TIFF
        """
        metadata = {}
        
        try:
            # Try FEI/ThermoFisher metadata
            if hasattr(tif, 'fei_metadata'):
                fei_meta = tif.fei_metadata
                if 'Scan' in fei_meta:
                    metadata['pixel_size_nm'] = fei_meta['Scan'].get('PixelWidth', 1.0) * 1e9
                    metadata['voltage_kv'] = fei_meta['Beam'].get('HV', 0) / 1000
                    metadata['magnification'] = fei_meta['Scan'].get('Magnification', 0)
            
            # Try Zeiss metadata
            elif hasattr(tif, 'sem_metadata'):
                sem_meta = tif.sem_metadata
                metadata['pixel_size_nm'] = sem_meta.get('PixelWidth', 1.0) * 1e9
                metadata['voltage_kv'] = sem_meta.get('EHT', 0) / 1000
                metadata['magnification'] = sem_meta.get('Mag', 0)
            
            # Try ImageJ metadata
            elif hasattr(tif, 'imagej_metadata'):
                ij_meta = tif.imagej_metadata
                if 'spacing' in ij_meta:
                    metadata['pixel_size_nm'] = ij_meta['spacing'] * 1e9
            
            # Standard TIFF tags
            tags = tif.pages[0].tags
            if 'XResolution' in tags and 'YResolution' in tags:
                x_res = tags['XResolution'].value
                if isinstance(x_res, tuple):
                    x_res = x_res[0] / x_res[1]
                # Convert to nm (assuming resolution in pixels per cm)
                metadata['pixel_size_nm'] = 1e7 / x_res if x_res > 0 else 1.0
            
        except Exception as e:
            print(f"Warning: Could not extract metadata: {e}")
        
        return metadata

class JPEGProcessor:
    """
    Handles JPEG images with overlay detection and removal
    """
    
    def __init__(self, use_visualization=False, debug_dir="debug_output"):
        self.overlay_detector = OverlayDetector(use_visualization, debug_dir)
        self.artifact_reducer = JPEGArtifactReducer()
        
    def process_jpeg_sem(self, image_path):
        """
        Complete JPEG processing pipeline
        """
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect JPEG quality
        jpeg_quality = self.artifact_reducer.detect_jpeg_quality(image)
        
        # Detect and remove overlays
        overlays = self.overlay_detector.detect_all_overlays(image)
        
        if overlays:
            image = self.overlay_detector.remove_overlays(image, overlays)
        
        # Reduce JPEG artifacts if needed
        if jpeg_quality < 85:
            image = self.artifact_reducer.reduce_artifacts(image, jpeg_quality)
        
        return image, overlays, jpeg_quality

class OverlayDetector:
    """
    Detects and removes text overlays, timestamps, and measurement annotations
    """
    
    def __init__(self, use_visualization=False, debug_dir="debug_output"):
        self.use_visualization = use_visualization
        self.debug_dir = debug_dir
        if use_visualization:
            os.makedirs(debug_dir, exist_ok=True)
    
    def detect_all_overlays(self, image):
        """
        Comprehensive overlay detection using multiple methods
        """
        overlays = []
        
        # Method 1: OCR text detection
        try:
            text_regions = self._detect_text_regions(image)
            overlays.extend(text_regions)
        except Exception as e:
            print(f"OCR detection failed: {e}")
        
        # Method 2: High contrast overlay detection
        contrast_regions = self._detect_high_contrast_overlays(image)
        overlays.extend(contrast_regions)
        
        # Method 3: Geometric pattern detection
        geometric_regions = self._detect_geometric_overlays(image)
        overlays.extend(geometric_regions)
        
        # Method 4: Corner overlay detection
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
        
        # Use Tesseract to detect text
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Confidence threshold
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text = data['text'][i].strip()
                if w > 10 and h > 10 and len(text) > 0:  # Size and content threshold
                    text_regions.append({
                        'type': 'text',
                        'bbox': (x, y, w, h),
                        'confidence': data['conf'][i],
                        'text': text
                    })
        
        return text_regions
    
    def _detect_high_contrast_overlays(self, image):
        """
        Detect overlays based on extreme pixel values
        """
        overlays = []
        
        # Detect very bright and dark regions
        white_mask = image > 240
        dark_mask = image < 15
        overlay_mask = white_mask | dark_mask
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        overlay_mask = cv2.morphologyEx(overlay_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        contours, _ = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            if 50 < area < 5000 and w > 10 and h > 8:
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
        
        # Edge detection and line detection
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=5)
        
        if lines is not None:
            # Group parallel lines for scale bars
            line_groups = self._group_parallel_lines(lines)
            
            for group in line_groups:
                if len(group) >= 2:
                    # Calculate bounding box
                    all_points = np.array(group).reshape(-1, 2)
                    x_min, y_min = np.min(all_points, axis=0)
                    x_max, y_max = np.max(all_points, axis=0)
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Check for scale bar patterns
                    if (width > 50 and height < 30) or (height > 50 and width < 30):
                        overlays.append({
                            'type': 'scale_bar',
                            'bbox': (x_min-5, y_min-5, width+10, height+10),
                            'confidence': 70,
                            'orientation': 'horizontal' if width > height else 'vertical'
                        })
        
        return overlays
    
    def _detect_corner_overlays(self, image):
        """
        Detect overlays in image corners (timestamps, logos)
        """
        overlays = []
        h, w = image.shape
        
        # Define corner regions (15% from each corner)
        corner_size_h = int(h * 0.15)
        corner_size_w = int(w * 0.15)
        
        corners = [
            (0, 0, corner_size_w, corner_size_h, "top_left"),
            (w-corner_size_w, 0, corner_size_w, corner_size_h, "top_right"),
            (0, h-corner_size_h, corner_size_w, corner_size_h, "bottom_left"),
            (w-corner_size_w, h-corner_size_h, corner_size_w, corner_size_h, "bottom_right")
        ]
        
        for i, (x, y, cw, ch, name) in enumerate(corners):
            corner_region = image[y:y+ch, x:x+cw]
            
            if self._has_corner_overlay(corner_region):
                overlays.append({
                    'type': 'corner_overlay',
                    'bbox': (x, y, cw, ch),
                    'confidence': 60,
                    'corner': i,
                    'corner_name': name
                })
        
        return overlays
    
    def _is_text_like_region(self, region):
        """
        Check if region has text-like characteristics
        """
        if region.size == 0:
            return False
        
        # Calculate edge density and contrast
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        contrast = region.std()
        
        return edge_density > 0.02 and contrast > 30
    
    def _has_corner_overlay(self, corner_region):
        """
        Check if corner region contains overlay information
        """
        if corner_region.size == 0:
            return False
        
        # Check for bimodal distribution (text on background)
        hist = cv2.calcHist([corner_region], [0], None, [256], [0, 256])
        
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 50:
                peaks.append(i)
        
        # Check for extreme values and high contrast
        white_pixels = np.sum(corner_region > 240)
        black_pixels = np.sum(corner_region < 15)
        total_pixels = corner_region.size
        extreme_ratio = (white_pixels + black_pixels) / total_pixels
        
        return len(peaks) >= 2 or extreme_ratio > 0.1
    
    def _group_parallel_lines(self, lines):
        """
        Group parallel lines for scale bar detection
        """
        if lines is None:
            return []
        
        line_params = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            line_params.append((angle, line[0]))
        
        # Group lines with similar angles
        angle_threshold = 0.2
        groups = []
        used = set()
        
        for i, (angle1, line1) in enumerate(line_params):
            if i in used:
                continue
            
            group = [line1]
            used.add(i)
            
            for j, (angle2, line2) in enumerate(line_params):
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
        
        # Sort by area
        overlays.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        
        merged = []
        for overlay in overlays:
            x, y, w, h = overlay['bbox']
            overlapped = False
            
            for merged_overlay in merged:
                mx, my, mw, mh = merged_overlay['bbox']
                
                # Check overlap
                if (x < mx + mw and x + w > mx and 
                    y < my + mh and y + h > my):
                    
                    # Calculate IoU
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
            # Add padding
            padding = 3
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            mask[y:y+h, x:x+w] = 255
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Use inpainting
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return result

class JPEGArtifactReducer:
    """
    Reduces JPEG compression artifacts
    """
    
    def detect_jpeg_quality(self, image):
        """
        Estimate JPEG quality based on blocking artifacts
        """
        h, w = image.shape
        block_size = 8
        
        blocking_score = 0
        block_count = 0
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = image[y:y+block_size, x:x+block_size].astype(np.float32)
                
                # Check for blocking artifacts at boundaries
                if x + block_size < w:
                    left_edge = block[:, -1]
                    right_block = image[y:y+block_size, x+block_size:x+2*block_size]
                    if right_block.shape[1] > 0:
                        right_edge = right_block[:, 0]
                        h_diff = np.mean(np.abs(left_edge - right_edge))
                        blocking_score += h_diff
                        block_count += 1
                
                if y + block_size < h:
                    top_edge = block[-1, :]
                    bottom_block = image[y+block_size:y+2*block_size, x:x+block_size]
                    if bottom_block.shape[0] > 0:
                        bottom_edge = bottom_block[0, :]
                        v_diff = np.mean(np.abs(top_edge - bottom_edge))
                        blocking_score += v_diff
                        block_count += 1
        
        avg_blocking = blocking_score / block_count if block_count > 0 else 0
        
        # Convert to quality estimate
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
            return image
        
        result = image.copy().astype(np.float32)
        
        # Bilateral filtering for blocking artifacts
        if estimated_quality < 70:
            result = cv2.bilateralFilter(result.astype(np.uint8), d=5, 
                                       sigmaColor=30, sigmaSpace=30).astype(np.float32)
        
        # Wavelet denoising for moderate artifacts
        if estimated_quality < 60:
            result = denoise_wavelet(result/255.0, sigma=0.1, 
                                   wavelet='db4', mode='soft') * 255
        
        # Edge-preserving filter for severe artifacts
        if estimated_quality < 40:
            result = cv2.edgePreservingFilter(result.astype(np.uint8), 
                                            flags=2, sigma_s=50, sigma_r=0.4)
        
        return result.astype(np.uint8)

class QualityAnalyzer:
    """
    Comprehensive image quality analysis
    """
    
    def calculate_comprehensive_metrics(self, image):
        """
        Calculate all quality metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['contrast'] = self._calculate_rms_contrast(image)
        metrics['brightness'] = self._calculate_brightness(image)
        metrics['sharpness'] = self._calculate_sharpness(image)
        
        # Advanced metrics
        snr_results = self._calculate_snr(image)
        metrics.update(snr_results)
        
        # Focus quality
        metrics['focus_quality'] = self._calculate_focus_quality(image)
        
        # Noise analysis
        metrics['noise_std'] = self._estimate_noise_std(image)
        
        # Overall quality score
        metrics['overall_quality'] = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _calculate_rms_contrast(self, image):
        """Root Mean Square contrast"""
        return float(np.std(image.astype(np.float64)))
    
    def _calculate_brightness(self, image):
        """Average brightness"""
        return float(np.mean(image))
    
    def _calculate_sharpness(self, image):
        """Laplacian variance sharpness"""
        laplacian = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
        return float(laplacian.var())
    
    def _calculate_snr(self, image):
        """Signal-to-noise ratio calculation"""
        # Local standard deviation method
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        noise_std = np.std(image - local_mean)
        
        signal_mean = np.mean(image)
        signal_max = np.max(image)
        
        snr_mean = 20 * np.log10(signal_mean / noise_std) if noise_std > 0 else float('inf')
        psnr = 20 * np.log10(signal_max / noise_std) if noise_std > 0 else float('inf')
        
        return {
            'snr_db': float(snr_mean),
            'psnr_db': float(psnr),
            'signal_mean': float(signal_mean),
            'signal_max': float(signal_max)
        }
    
    def _calculate_focus_quality(self, image):
        """FFT-based focus quality"""
        # High frequency content indicates good focus
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        rows, cols = image.shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        mask = (x - center_col)**2 + (y - center_row)**2 > (min(rows, cols) * 0.1)**2
        
        high_freq_energy = np.sum(magnitude_spectrum[mask]**2)
        total_energy = np.sum(magnitude_spectrum**2)
        
        return float(high_freq_energy / total_energy) if total_energy > 0 else 0
    
    def _estimate_noise_std(self, image):
        """Estimate noise standard deviation"""
        # High-frequency noise estimation using Laplacian
        laplacian = cv2.Laplacian(image.astype(np.float64), cv2.CV_64F)
        noise_std = np.std(laplacian) / np.sqrt(6)
        return float(noise_std)
    
    def _calculate_overall_score(self, metrics):
        """Calculate overall quality score (0-100)"""
        # Normalize and combine metrics
        contrast_score = min(100, metrics['contrast'] / 50 * 100)
        sharpness_score = min(100, metrics['sharpness'] / 1000 * 100)
        focus_score = min(100, metrics['focus_quality'] * 1000)
        snr_score = min(100, max(0, metrics['snr_db'] - 10) * 2)
        
        overall = (contrast_score + sharpness_score + focus_score + snr_score) / 4
        return min(100, max(0, overall))

class PatternAnalyzer:
    """
    Analyzes semiconductor patterns (lines, contacts)
    """
    
    def analyze_patterns(self, image, pixel_size_nm):
        """
        Comprehensive pattern analysis
        """
        results = {
            'line_space_analysis': self._analyze_line_space_patterns(image, pixel_size_nm),
            'contact_hole_analysis': self._analyze_contact_holes(image, pixel_size_nm),
            'pattern_type': self._detect_pattern_type(image)
        }
        
        return results
    
    def _analyze_line_space_patterns(self, image, pixel_size_nm):
        """
        Analyze line/space patterns for CD measurement
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        measurements = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                # Fit bounding rectangle
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                # Calculate CD (assuming line width is the smaller dimension)
                cd_pixels = min(width, height)
                cd_nm = cd_pixels * pixel_size_nm
                
                # Calculate LER (Line Edge Roughness)
                ler = self._calculate_ler(contour, pixel_size_nm)
                
                measurements.append({
                    'cd_nm': float(cd_nm),
                    'ler_3sigma_nm': float(ler),
                    'area_pixels': float(cv2.contourArea(contour))
                })
        
        return {
            'measurements': measurements,
            'average_cd_nm': float(np.mean([m['cd_nm'] for m in measurements])) if measurements else 0,
            'cd_uniformity': float(np.std([m['cd_nm'] for m in measurements])) if measurements else 0
        }
    
    def _analyze_contact_holes(self, image, pixel_size_nm):
        """
        Analyze contact holes for diameter and circularity
        """
        # Threshold and find circular features
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hole_measurements = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                if len(contour) >= 5:
                    # Fit ellipse
                    ellipse = cv2.fitEllipse(contour)
                    center, (width, height), angle = ellipse
                    
                    # Calculate metrics
                    diameter_nm = ((width + height) / 2) * pixel_size_nm
                    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True)**2)
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    hole_measurements.append({
                        'diameter_nm': float(diameter_nm),
                        'circularity': float(circularity),
                        'aspect_ratio': float(aspect_ratio)
                    })
        
        return {
            'measurements': hole_measurements,
            'average_diameter_nm': float(np.mean([m['diameter_nm'] for m in hole_measurements])) if hole_measurements else 0,
            'circularity_avg': float(np.mean([m['circularity'] for m in hole_measurements])) if hole_measurements else 0
        }
    
    def _calculate_ler(self, contour, pixel_size_nm):
        """
        Calculate Line Edge Roughness
        """
        if len(contour) < 10:
            return 0
        
        # Extract edge coordinates
        edge_points = contour.reshape(-1, 2)
        
        # Fit polynomial to edge
        if len(edge_points) > 3:
            try:
                poly = np.polyfit(edge_points[:, 0], edge_points[:, 1], 3)
                fitted_edge = np.polyval(poly, edge_points[:, 0])
                
                # Calculate deviations
                deviations = (edge_points[:, 1] - fitted_edge) * pixel_size_nm
                
                # 3σ LER
                ler_3sigma = 3 * np.std(deviations)
                return ler_3sigma
            except:
                return 0
        
        return 0
    
    def _detect_pattern_type(self, image):
        """
        Detect if image contains lines, contacts, or other patterns
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'unknown'
        
        # Analyze contour shapes
        circular_count = 0
        linear_count = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                # Calculate shape metrics
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
                
                # Fit bounding rectangle
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
                
                if circularity > 0.7:  # Circular
                    circular_count += 1
                elif aspect_ratio > 3:  # Linear
                    linear_count += 1
        
        if circular_count > linear_count:
            return 'contact_holes'
        elif linear_count > circular_count:
            return 'line_space'
        else:
            return 'mixed'

class DriftCompensation:
    """
    Handles stage drift compensation and quality metric adjustment
    """
    
    def __init__(self):
        self.drift_history = []
    
    def analyze_drift(self, current_image, reference_image):
        """
        Analyze drift between current and reference images
        """
        # Register images using phase correlation
        shift = self._register_images_phase_correlation(current_image, reference_image)
        
        # Calculate drift magnitude
        drift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)
        
        # Store in history
        drift_info = {
            'shift_x': float(shift[0]),
            'shift_y': float(shift[1]),
            'drift_magnitude': float(drift_magnitude),
            'timestamp': datetime.now().isoformat()
        }
        
        self.drift_history.append(drift_info)
        
        return drift_info
    
    def _register_images_phase_correlation(self, img1, img2):
        """
        Register images using phase correlation
        """
        # Ensure same size
        if img1.shape != img2.shape:
            min_h = min(img1.shape[0], img2.shape[0])
            min_w = min(img1.shape[1], img2.shape[1])
            img1 = img1[:min_h, :min_w]
            img2 = img2[:min_h, :min_w]
        
        # Apply window function
        window = np.hanning(img1.shape[0])[:, None] * np.hanning(img1.shape[1])[None, :]
        img1_windowed = img1.astype(np.float32) * window
        img2_windowed = img2.astype(np.float32) * window
        
        # Phase correlation
        shift = cv2.phaseCorrelate(img1_windowed, img2_windowed)[0]
        
        return shift
    
    def compensate_quality_metrics(self, metrics, drift_magnitude):
        """
        Adjust quality metrics based on drift magnitude
        """
        if drift_magnitude < 2.0:  # Minor drift
            return metrics
        
        # Apply compensation factors for drift
        compensation_factor = max(0.8, 1.0 - (drift_magnitude - 2.0) * 0.02)
        
        compensated_metrics = metrics.copy()
        compensated_metrics['sharpness'] *= compensation_factor
        compensated_metrics['focus_quality'] *= compensation_factor
        
        return compensated_metrics

# Example usage and testing functions
def create_test_images():
    """
    Create test images for demonstration
    """
    # Create TIFF-like image
    tiff_image = np.random.randint(80, 180, (1024, 1024), dtype=np.uint8)
    cv2.circle(tiff_image, (300, 300), 80, 200, -1)
    cv2.rectangle(tiff_image, (500, 400), (650, 500), 120, -1)
    cv2.imwrite('test_sem.tif', tiff_image)
    
    # Create JPEG with overlays
    jpeg_image = tiff_image.copy()
    cv2.putText(jpeg_image, "2024-07-02 14:30:25", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    cv2.putText(jpeg_image, "5.00 kV  x50,000", (20, 1000), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.putText(jpeg_image, "CD: 28.5nm", (750, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    # Add scale bar
    cv2.line(jpeg_image, (750, 950), (850, 950), 255, 4)
    cv2.putText(jpeg_image, "100nm", (760, 940), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    cv2.imwrite('test_sem_overlay.jpg', jpeg_image, 
                [cv2.IMWRITE_JPEG_QUALITY, 75])

def demo_analysis():
    """
    Demonstrate the complete analysis system
    """
    print("🔬 CD-SEM Image Quality Analysis Demo")
    print("=" * 50)
    
    # Create test images
    create_test_images()
    
    # Initialize analyzer
    analyzer = CDSEMImageAnalyzer(use_visualization=True)
    
    # Analyze TIFF image
    print("\n📊 Analyzing TIFF image...")
    tiff_result = analyzer.analyze_image('test_sem.tif')
    
    print(f"✅ TIFF Analysis Complete:")
    print(f"   📈 Overall Quality: {tiff_result['quality_metrics']['overall_quality']:.1f}/100")
    print(f"   🔍 Contrast: {tiff_result['quality_metrics']['contrast']:.1f}")
    print(f"   ⚡ Sharpness: {tiff_result['quality_metrics']['sharpness']:.1f}")
    print(f"   📡 SNR: {tiff_result['quality_metrics']['snr_db']:.1f} dB")
    
    # Analyze JPEG image
    print("\n📊 Analyzing JPEG image with overlays...")
    jpeg_result = analyzer.analyze_image('test_sem_overlay.jpg', remove_overlays=True)
    
    print(f"✅ JPEG Analysis Complete:")
    print(f"   📈 Overall Quality: {jpeg_result['quality_metrics']['overall_quality']:.1f}/100")
    print(f"   🎯 Overlays Removed: {len(jpeg_result['overlays_removed']) if jpeg_result['overlays_removed'] else 0}")
    print(f"   📐 Analysis Area: {jpeg_result['analysis_area_percentage']:.1f}%")
    print(f"   🔍 Contrast: {jpeg_result['quality_metrics']['contrast']:.1f}")
    print(f"   ⚡ Sharpness: {jpeg_result['quality_metrics']['sharpness']:.1f}")
    
    # Save results
    with open('analysis_results.json', 'w') as f:
        json.dump({
            'tiff_analysis': tiff_result,
            'jpeg_analysis': jpeg_result
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: analysis_results.json")
    print(f"🎨 Debug images saved to: debug_output/")
    
    return tiff_result, jpeg_result

if __name__ == "__main__":
    # Run demonstration
    try:
        demo_analysis()
        print("\n🎉 Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
