import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFont, ImageDraw
import pytesseract
from skimage import restoration, morphology, filters
import os
from datetime import datetime

class VisualOverlayDetector:
    """
    Enhanced overlay detector with comprehensive visualization capabilities
    """
    
    def __init__(self, save_debug_images=True, debug_dir="debug_output"):
        self.save_debug_images = save_debug_images
        self.debug_dir = debug_dir
        self.step_counter = 0
        
        # Create debug directory
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)
            
        # Colors for different overlay types
        self.colors = {
            'text': (255, 0, 0),      # Red
            'high_contrast': (0, 255, 0),  # Green
            'scale_bar': (0, 0, 255),     # Blue
            'corner_overlay': (255, 255, 0),  # Yellow
            'geometric': (255, 0, 255)    # Magenta
        }
    
    def analyze_with_visualization(self, image_path):
        """
        Complete analysis with step-by-step visualization
        """
        print(f"🔍 Starting visual analysis of: {os.path.basename(image_path)}")
        print("=" * 60)
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        self.step_counter = 0
        
        # Step 0: Show original image
        self._save_step_image(original_image, "00_original_image", 
                             "Original JPEG SEM Image")
        
        # Initialize results tracking
        all_overlays = []
        detection_results = {}
        
        print("\n🎯 Detection Methods:")
        print("-" * 30)
        
        # Method 1: OCR Text Detection
        print("1️⃣  OCR Text Detection...")
        text_overlays, text_debug = self._detect_text_with_visualization(image)
        all_overlays.extend(text_overlays)
        detection_results['ocr_text'] = {
            'overlays': text_overlays,
            'debug_info': text_debug
        }
        print(f"   ✅ Found {len(text_overlays)} text regions")
        
        # Method 2: High Contrast Detection
        print("2️⃣  High Contrast Overlay Detection...")
        contrast_overlays, contrast_debug = self._detect_high_contrast_with_visualization(image)
        all_overlays.extend(contrast_overlays)
        detection_results['high_contrast'] = {
            'overlays': contrast_overlays,
            'debug_info': contrast_debug
        }
        print(f"   ✅ Found {len(contrast_overlays)} high contrast regions")
        
        # Method 3: Geometric Pattern Detection
        print("3️⃣  Geometric Pattern Detection...")
        geometric_overlays, geometric_debug = self._detect_geometric_with_visualization(image)
        all_overlays.extend(geometric_overlays)
        detection_results['geometric'] = {
            'overlays': geometric_overlays,
            'debug_info': geometric_debug
        }
        print(f"   ✅ Found {len(geometric_overlays)} geometric patterns")
        
        # Method 4: Corner Overlay Detection
        print("4️⃣  Corner Overlay Detection...")
        corner_overlays, corner_debug = self._detect_corner_overlays_with_visualization(image)
        all_overlays.extend(corner_overlays)
        detection_results['corner'] = {
            'overlays': corner_overlays,
            'debug_info': corner_debug
        }
        print(f"   ✅ Found {len(corner_overlays)} corner overlays")
        
        # Merge overlapping regions
        print("\n🔄 Merging overlapping regions...")
        merged_overlays = self._merge_overlapping_regions(all_overlays)
        print(f"   📊 {len(all_overlays)} regions → {len(merged_overlays)} after merging")
        
        # Show all detections combined
        self._visualize_all_detections(image, detection_results, merged_overlays)
        
        # Create removal mask and show final result
        print("\n🧹 Applying overlay removal...")
        cleaned_image, removal_mask = self._remove_overlays_with_visualization(image, merged_overlays)
        
        # Final comparison
        self._create_before_after_comparison(original_image, cleaned_image, merged_overlays)
        
        print(f"\n✨ Analysis complete! Debug images saved to: {self.debug_dir}")
        print(f"📈 Quality assessment area: {self._calculate_analysis_area_percentage(image.shape, removal_mask):.1f}% of image")
        
        return cleaned_image, merged_overlays, detection_results
    
    def _detect_text_with_visualization(self, image):
        """
        OCR-based text detection with visualization
        """
        overlays = []
        debug_info = {}
        
        try:
            # Create visualization image
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Use OCR to detect text
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Analyze OCR results
            confidence_threshold = 30
            size_threshold = (10, 8)  # min width, height
            
            detected_texts = []
            
            for i in range(len(data['text'])):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence > confidence_threshold and len(text) > 0:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    if w >= size_threshold[0] and h >= size_threshold[1]:
                        # Draw detection on visualization
                        cv2.rectangle(vis_image, (x, y), (x+w, y+h), self.colors['text'], 2)
                        cv2.putText(vis_image, f"{confidence}%", (x, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
                        
                        overlay = {
                            'type': 'text',
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'text': text
                        }
                        overlays.append(overlay)
                        detected_texts.append(f"'{text}' ({confidence}%)")
            
            # Save visualization
            self._save_step_image(vis_image, "01_ocr_text_detection", 
                                 f"OCR Text Detection (Found: {len(overlays)})")
            
            # Create detailed OCR analysis
            self._create_ocr_analysis_image(image, data, confidence_threshold)
            
            debug_info = {
                'total_ocr_regions': len(data['text']),
                'valid_detections': len(overlays),
                'confidence_threshold': confidence_threshold,
                'detected_texts': detected_texts
            }
            
        except Exception as e:
            print(f"   ⚠️  OCR detection failed: {e}")
            self._save_step_image(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 
                                 "01_ocr_text_detection", "OCR Detection Failed")
            debug_info = {'error': str(e)}
        
        return overlays, debug_info
    
    def _detect_high_contrast_with_visualization(self, image):
        """
        High contrast overlay detection with visualization
        """
        overlays = []
        
        # Create multi-step visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('High Contrast Overlay Detection Process', fontsize=16)
        
        # Step 1: Original image
        axes[0,0].imshow(image, cmap='gray')
        axes[0,0].set_title('Original Image')
        axes[0,0].axis('off')
        
        # Step 2: White regions (bright text)
        white_threshold = 240
        white_mask = image > white_threshold
        axes[0,1].imshow(white_mask, cmap='gray')
        axes[0,1].set_title(f'White Regions (>{white_threshold})')
        axes[0,1].axis('off')
        
        # Step 3: Dark regions (dark text)
        dark_threshold = 15
        dark_mask = image < dark_threshold
        axes[0,2].imshow(dark_mask, cmap='gray')
        axes[0,2].set_title(f'Dark Regions (<{dark_threshold})')
        axes[0,2].axis('off')
        
        # Step 4: Combined mask
        combined_mask = white_mask | dark_mask
        axes[1,0].imshow(combined_mask, cmap='gray')
        axes[1,0].set_title('Combined Mask')
        axes[1,0].axis('off')
        
        # Step 5: Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        axes[1,1].imshow(cleaned_mask, cmap='gray')
        axes[1,1].set_title('After Morphological Cleaning')
        axes[1,1].axis('off')
        
        # Step 6: Final detections
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Find contours and analyze
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_regions = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter criteria
            min_area = 50
            max_area = 5000
            min_width = 10
            min_height = 8
            
            if (min_area < area < max_area and w > min_width and h > min_height):
                # Additional validation: check if region looks like text
                region = image[y:y+h, x:x+w]
                if self._is_text_like_region(region):
                    # Draw detection
                    cv2.rectangle(vis_image, (x, y), (x+w, y+h), self.colors['high_contrast'], 2)
                    cv2.putText(vis_image, f"A:{int(area)}", (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['high_contrast'], 1)
                    
                    overlays.append({
                        'type': 'high_contrast',
                        'bbox': (x, y, w, h),
                        'confidence': 80,
                        'area': area
                    })
                    valid_regions += 1
        
        axes[1,2].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[1,2].set_title(f'Final Detections ({valid_regions})')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/02_high_contrast_process.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save final detection image
        self._save_step_image(vis_image, "02_high_contrast_detection", 
                             f"High Contrast Detection (Found: {len(overlays)})")
        
        debug_info = {
            'white_threshold': white_threshold,
            'dark_threshold': dark_threshold,
            'total_contours': len(contours),
            'valid_regions': len(overlays),
            'filtering_criteria': {
                'min_area': min_area,
                'max_area': max_area,
                'min_dimensions': (min_width, min_height)
            }
        }
        
        return overlays, debug_info
    
    def _detect_geometric_with_visualization(self, image):
        """
        Geometric pattern detection with visualization
        """
        overlays = []
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Geometric Pattern Detection (Scale Bars & Measurements)', fontsize=14)
        
        # Step 1: Edge detection
        edges = cv2.Canny(image, 50, 150)
        axes[0,0].imshow(edges, cmap='gray')
        axes[0,0].set_title('Edge Detection')
        axes[0,0].axis('off')
        
        # Step 2: Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=5)
        
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 255), 1)
                line_count += 1
        
        axes[0,1].imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
        axes[0,1].set_title(f'Detected Lines ({line_count})')
        axes[0,1].axis('off')
        
        # Step 3: Group parallel lines
        if lines is not None:
            line_groups = self._group_parallel_lines(lines)
            
            group_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            colors_for_groups = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
            
            for i, group in enumerate(line_groups):
                color = colors_for_groups[i % len(colors_for_groups)]
                for line in group:
                    x1, y1, x2, y2 = line
                    cv2.line(group_image, (x1, y1), (x2, y2), color, 2)
            
            axes[1,0].imshow(cv2.cvtColor(group_image, cv2.COLOR_BGR2RGB))
            axes[1,0].set_title(f'Grouped Parallel Lines ({len(line_groups)} groups)')
            axes[1,0].axis('off')
        else:
            axes[1,0].imshow(image, cmap='gray')
            axes[1,0].set_title('No Lines Detected')
            axes[1,0].axis('off')
            line_groups = []
        
        # Step 4: Identify scale bars and measurement marks
        detection_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        for group in line_groups:
            if len(group) >= 2:  # At least 2 parallel lines
                # Calculate bounding box
                all_points = np.array(group).reshape(-1, 2)
                x_min, y_min = np.min(all_points, axis=0)
                x_max, y_max = np.max(all_points, axis=0)
                
                width = x_max - x_min
                height = y_max - y_min
                
                # Check for scale bar patterns
                is_scale_bar = False
                orientation = None
                
                if width > 50 and height < 30:  # Horizontal scale bar
                    is_scale_bar = True
                    orientation = 'horizontal'
                elif height > 50 and width < 30:  # Vertical scale bar
                    is_scale_bar = True
                    orientation = 'vertical'
                
                if is_scale_bar:
                    # Add padding
                    x_min = max(0, x_min - 5)
                    y_min = max(0, y_min - 5)
                    width = min(image.shape[1] - x_min, width + 10)
                    height = min(image.shape[0] - y_min, height + 10)
                    
                    # Draw detection
                    cv2.rectangle(detection_image, (x_min, y_min), 
                                (x_min + width, y_min + height), 
                                self.colors['scale_bar'], 2)
                    cv2.putText(detection_image, orientation[:4], 
                              (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.4, self.colors['scale_bar'], 1)
                    
                    overlays.append({
                        'type': 'scale_bar',
                        'bbox': (x_min, y_min, width, height),
                        'confidence': 70,
                        'orientation': orientation,
                        'line_count': len(group)
                    })
        
        axes[1,1].imshow(cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title(f'Scale Bar Detection ({len(overlays)})')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/03_geometric_process.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save final detection
        self._save_step_image(detection_image, "03_geometric_detection", 
                             f"Geometric Pattern Detection (Found: {len(overlays)})")
        
        debug_info = {
            'edges_detected': np.sum(edges > 0),
            'lines_detected': line_count,
            'line_groups': len(line_groups),
            'scale_bars_found': len(overlays)
        }
        
        return overlays, debug_info
    
    def _detect_corner_overlays_with_visualization(self, image):
        """
        Corner overlay detection with visualization
        """
        overlays = []
        h, w = image.shape
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Corner Overlay Detection (Timestamps & Logos)', fontsize=14)
        
        # Show original with corner regions marked
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Define corner regions (15% from each corner)
        corner_size_h = int(h * 0.15)
        corner_size_w = int(w * 0.15)
        
        corners = [
            (0, 0, corner_size_w, corner_size_h, "Top-Left"),
            (w-corner_size_w, 0, corner_size_w, corner_size_h, "Top-Right"),
            (0, h-corner_size_h, corner_size_w, corner_size_h, "Bottom-Left"),
            (w-corner_size_w, h-corner_size_h, corner_size_w, corner_size_h, "Bottom-Right")
        ]
        
        # Draw corner regions
        for i, (x, y, cw, ch, name) in enumerate(corners):
            cv2.rectangle(vis_image, (x, y), (x+cw, y+ch), (100, 100, 100), 1)
            cv2.putText(vis_image, name, (x+5, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        axes[0,0].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[0,0].set_title('Corner Regions Defined')
        axes[0,0].axis('off')
        
        # Analyze each corner
        corner_results = []
        
        for i, (x, y, cw, ch, name) in enumerate(corners):
            corner_region = image[y:y+ch, x:x+cw]
            
            # Analyze corner region
            has_overlay, analysis = self._analyze_corner_region(corner_region)
            
            # Visualize corner analysis
            row = i // 2
            col = (i % 2) + 1
            
            axes[row, col].imshow(corner_region, cmap='gray')
            axes[row, col].set_title(f'{name}\nOverlay: {"Yes" if has_overlay else "No"}')
            axes[row, col].axis('off')
            
            corner_results.append({
                'corner': i,
                'name': name,
                'has_overlay': has_overlay,
                'analysis': analysis
            })
            
            if has_overlay:
                overlays.append({
                    'type': 'corner_overlay',
                    'bbox': (x, y, cw, ch),
                    'confidence': 60,
                    'corner': i,
                    'corner_name': name,
                    'analysis': analysis
                })
        
        # Show detection results
        detection_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for overlay in overlays:
            x, y, cw, ch = overlay['bbox']
            cv2.rectangle(detection_image, (x, y), (x+cw, y+ch), 
                         self.colors['corner_overlay'], 2)
            cv2.putText(detection_image, overlay['corner_name'][:6], 
                       (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, self.colors['corner_overlay'], 1)
        
        axes[1,2].imshow(cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB))
        axes[1,2].set_title(f'Detected Corner Overlays ({len(overlays)})')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/04_corner_process.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save final detection
        self._save_step_image(detection_image, "04_corner_detection", 
                             f"Corner Overlay Detection (Found: {len(overlays)})")
        
        debug_info = {
            'corner_size_percent': 15,
            'corners_analyzed': len(corners),
            'corner_results': corner_results
        }
        
        return overlays, debug_info
    
    def _analyze_corner_region(self, corner_region):
        """
        Analyze if corner region contains overlay information
        """
        if corner_region.size == 0:
            return False, {}
        
        analysis = {}
        
        # 1. Histogram analysis for bimodal distribution
        hist = cv2.calcHist([corner_region], [0], None, [256], [0, 256])
        
        # Find peaks in histogram
        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 50:
                peaks.append(i)
        
        analysis['histogram_peaks'] = len(peaks)
        analysis['has_bimodal'] = len(peaks) >= 2
        
        # 2. Edge density analysis
        edges = cv2.Canny(corner_region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        analysis['edge_density'] = edge_density
        analysis['high_edge_density'] = edge_density > 0.02
        
        # 3. Contrast analysis
        contrast = corner_region.std()
        analysis['contrast'] = contrast
        analysis['high_contrast'] = contrast > 30
        
        # 4. Check for extreme values (common in overlays)
        white_pixels = np.sum(corner_region > 240)
        black_pixels = np.sum(corner_region < 15)
        total_pixels = corner_region.size
        
        analysis['white_pixel_ratio'] = white_pixels / total_pixels
        analysis['black_pixel_ratio'] = black_pixels / total_pixels
        analysis['has_extreme_values'] = (white_pixels + black_pixels) / total_pixels > 0.1
        
        # Decision: overlay detected if multiple criteria met
        criteria_met = sum([
            analysis['has_bimodal'],
            analysis['high_edge_density'],
            analysis['high_contrast'],
            analysis['has_extreme_values']
        ])
        
        has_overlay = criteria_met >= 2
        analysis['criteria_met'] = criteria_met
        
        return has_overlay, analysis
    
    def _visualize_all_detections(self, image, detection_results, merged_overlays):
        """
        Create comprehensive visualization of all detection methods
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Complete Overlay Detection Summary', fontsize=16)
        
        # Individual method results
        methods = [
            ('ocr_text', 'OCR Text Detection'),
            ('high_contrast', 'High Contrast Detection'),
            ('geometric', 'Geometric Patterns'),
            ('corner', 'Corner Overlays')
        ]
        
        for i, (method_key, method_name) in enumerate(methods):
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlays = detection_results[method_key]['overlays']
            
            for overlay in overlays:
                x, y, w, h = overlay['bbox']
                color = self.colors.get(overlay['type'], (128, 128, 128))
                cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            row = i // 2
            col = i % 2
            axes[row, col].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'{method_name} ({len(overlays)})')
            axes[row, col].axis('off')
        
        # All detections combined (before merging)
        all_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        total_before = 0
        for method_key in detection_results:
            overlays = detection_results[method_key]['overlays']
            total_before += len(overlays)
            for overlay in overlays:
                x, y, w, h = overlay['bbox']
                color = self.colors.get(overlay['type'], (128, 128, 128))
                cv2.rectangle(all_image, (x, y), (x+w, y+h), color, 1)
        
        axes[1, 2].imshow(cv2.cvtColor(all_image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'All Detections Before Merge ({total_before})')
        axes[1, 2].axis('off')
        
        # Final merged result
        final_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for overlay in merged_overlays:
            x, y, w, h = overlay['bbox']
            color = self.colors.get(overlay['type'], (128, 128, 128))
            cv2.rectangle(final_image, (x, y), (x+w, y+h), color, 3)
            # Add confidence score
            cv2.putText(final_image, f"{overlay['confidence']}", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        axes[0, 2].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'Final Merged Result ({len(merged_overlays)})')
        axes[0, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/05_detection_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save final detection image
        self._save_step_image(final_image, "05_all_detections", 
                             f"All Detection Methods Combined")
    
    def _remove_overlays_with_visualization(self, image, overlays):
        """
        Remove overlays with detailed visualization of the process
        """
        if not overlays:
            return image, np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Create removal mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for overlay in overlays:
            x, y, w, h = overlay['bbox']
            # Add padding for complete removal
            padding = 3
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            mask[y:y+h, x:x+w] = 255
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply inpainting
        result = cv2.inpaint(image, mask_cleaned, 3, cv2.INPAINT_TELEA)
        
        # Create visualization of removal process
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('Overlay Removal Process', fontsize=14)
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Removal Mask')
        axes[1].axis('off')
        
        axes[2].imshow(mask_cleaned, cmap='gray')
        axes[2].set_title('Cleaned Mask')
        axes[2].axis('off')
        
        axes[3].imshow(result, cmap='gray')
        axes[3].set_title('After Inpainting')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/06_removal_process.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual steps
        self._save_step_image(mask, "06a_removal_mask", "Areas to be Removed")
        self._save_step_image(result, "06b_cleaned_image", "After Overlay Removal")
        
        return result, mask_cleaned
    
    def _create_before_after_comparison(self, original, cleaned, overlays):
        """
        Create side-by-side comparison with detailed statistics
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle('Before vs After Overlay Removal', fontsize=16)
        
        # Original image with overlays marked
        original_marked = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        for overlay in overlays:
            x, y, w, h = overlay['bbox']
            color = self.colors.get(overlay['type'], (255, 255, 255))
            cv2.rectangle(original_marked, (x, y), (x+w, y+h), color, 2)
        
        axes[0].imshow(cv2.cvtColor(original_marked, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Original with {len(overlays)} Overlays Detected')
        axes[0].axis('off')
        
        axes[1].imshow(cleaned, cmap='gray')
        axes[1].set_title('Cleaned Image (Overlays Removed)')
        axes[1].axis('off')
        
        # Add statistics
        total_overlay_area = sum(overlay['bbox'][2] * overlay['bbox'][3] for overlay in overlays)
        image_area = original.shape[0] * original.shape[1]
        overlay_percentage = (total_overlay_area / image_area) * 100
        
        stats_text = f"""
        Statistics:
        • Total overlays: {len(overlays)}
        • Overlay area: {overlay_percentage:.2f}% of image
        • Analysis area: {100-overlay_percentage:.2f}% of image
        • Image size: {original.shape[1]}×{original.shape[0]}
        """
        
        plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/07_before_after_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _create_ocr_analysis_image(self, image, ocr_data, confidence_threshold):
        """
        Create detailed OCR analysis visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Detailed OCR Analysis', fontsize=14)
        
        # Confidence distribution
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        axes[0,0].hist(confidences, bins=20, alpha=0.7, color='blue')
        axes[0,0].axvline(confidence_threshold, color='red', linestyle='--', 
                         label=f'Threshold: {confidence_threshold}')
        axes[0,0].set_title('OCR Confidence Distribution')
        axes[0,0].set_xlabel('Confidence %')
        axes[0,0].set_ylabel('Count')
        axes[0,0].legend()
        
        # Text length distribution
        text_lengths = [len(text.strip()) for text in ocr_data['text'] 
                       if len(text.strip()) > 0]
        if text_lengths:
            axes[0,1].hist(text_lengths, bins=10, alpha=0.7, color='green')
            axes[0,1].set_title('Detected Text Length Distribution')
            axes[0,1].set_xlabel('Text Length (characters)')
            axes[0,1].set_ylabel('Count')
        else:
            axes[0,1].text(0.5, 0.5, 'No text detected', ha='center', va='center')
            axes[0,1].set_title('Text Length Distribution')
        
        # All OCR regions (including low confidence)
        all_regions_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(len(ocr_data['text'])):
            conf = int(ocr_data['conf'][i])
            if conf > 0:
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                color = (0, 255, 0) if conf > confidence_threshold else (0, 0, 255)
                cv2.rectangle(all_regions_image, (x, y), (x+w, y+h), color, 1)
        
        axes[1,0].imshow(cv2.cvtColor(all_regions_image, cv2.COLOR_BGR2RGB))
        axes[1,0].set_title('All OCR Regions (Green: Valid, Red: Low confidence)')
        axes[1,0].axis('off')
        
        # Valid text regions only
        valid_regions_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        valid_count = 0
        for i in range(len(ocr_data['text'])):
            conf = int(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            if conf > confidence_threshold and len(text) > 0:
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                cv2.rectangle(valid_regions_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(valid_regions_image, f"{conf}%", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                valid_count += 1
        
        axes[1,1].imshow(cv2.cvtColor(valid_regions_image, cv2.COLOR_BGR2RGB))
        axes[1,1].set_title(f'Valid Text Regions ({valid_count})')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.debug_dir}/01b_ocr_analysis_detail.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_step_image(self, image, filename, title):
        """Save step image with consistent formatting"""
        self.step_counter += 1
        
        if len(image.shape) == 3:
            # Color image
            cv2.imwrite(f"{self.debug_dir}/{filename}.png", image)
        else:
            # Grayscale image
            cv2.imwrite(f"{self.debug_dir}/{filename}.png", image)
        
        print(f"   💾 Saved: {filename}.png - {title}")
    
    def _calculate_analysis_area_percentage(self, image_shape, removal_mask):
        """Calculate percentage of image available for quality analysis"""
        total_pixels = image_shape[0] * image_shape[1]
        removed_pixels = np.sum(removal_mask > 0)
        analysis_pixels = total_pixels - removed_pixels
        return (analysis_pixels / total_pixels) * 100
    
    # Helper methods (reuse from previous implementation)
    def _is_text_like_region(self, region):
        """Check if region has text-like characteristics"""
        if region.size == 0:
            return False
        
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        contrast = region.std()
        
        return edge_density > 0.02 and contrast > 30
    
    def _group_parallel_lines(self, lines):
        """Group parallel lines for scale bar detection"""
        if lines is None:
            return []
        
        line_params = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            line_params.append((angle, length, center, line[0]))
        
        angle_threshold = 0.2
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
        """Merge overlapping overlay regions"""
        if not overlays:
            return []
        
        overlays.sort(key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)
        
        merged = []
        for overlay in overlays:
            x, y, w, h = overlay['bbox']
            overlapped = False
            
            for merged_overlay in merged:
                mx, my, mw, mh = merged_overlay['bbox']
                
                if (x < mx + mw and x + w > mx and 
                    y < my + mh and y + h > my):
                    
                    intersection = max(0, min(x + w, mx + mw) - max(x, mx)) * \
                                  max(0, min(y + h, my + mh) - max(y, my))
                    union = w * h + mw * mh - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.3:
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

# Usage example
def test_visual_detection():
    """
    Test the visual detection system
    """
    # Create a test SEM image with overlays
    def create_test_image():
        # Create base image
        img = np.random.randint(80, 180, (800, 1000), dtype=np.uint8)
        
        # Add some SEM-like features
        cv2.circle(img, (300, 300), 80, 200, -1)
        cv2.rectangle(img, (500, 400), (650, 500), 120, -1)
        
        # Add text overlays
        cv2.putText(img, "2024-07-02 14:30:25", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        cv2.putText(img, "5.00 kV  x50,000  WD 4.0mm", (20, 770), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(img, "CD: 28.5nm", (750, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        
        # Add scale bar
        cv2.line(img, (750, 750), (850, 750), 255, 4)
        cv2.line(img, (750, 745), (750, 755), 255, 2)
        cv2.line(img, (850, 745), (850, 755), 255, 2)
        cv2.putText(img, "100nm", (760, 740), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        return img
    
    # Create and save test image
    test_img = create_test_image()
    
    # Save as JPEG with moderate compression
    cv2.imwrite('test_sem_overlay.jpg', test_img, 
                [cv2.IMWRITE_JPEG_QUALITY, 75])
    
    print("🧪 Testing Visual Overlay Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = VisualOverlayDetector(save_debug_images=True, debug_dir="overlay_debug")
    
    # Run analysis
    cleaned_image, overlays, detection_results = detector.analyze_with_visualization('test_sem_overlay.jpg')
    
    print(f"\n📊 Final Results:")
    print(f"   🎯 Total overlays detected: {len(overlays)}")
    
    overlay_types = {}
    for overlay in overlays:
        overlay_type = overlay['type']
        overlay_types[overlay_type] = overlay_types.get(overlay_type, 0) + 1
    
    for overlay_type, count in overlay_types.items():
        print(f"   📋 {overlay_type}: {count}")
    
    # Calculate quality metrics for comparison
    from skimage.metrics import structural_similarity as ssim
    
    original = cv2.imread('test_sem_overlay.jpg', cv2.IMREAD_GRAYSCALE)
    similarity = ssim(original, cleaned_image)
    
    print(f"\n📈 Image Quality Comparison:")
    print(f"   🔍 Structural similarity: {similarity:.3f}")
    print(f"   📐 Original contrast: {original.std():.1f}")
    print(f"   📐 Cleaned contrast: {cleaned_image.std():.1f}")
    
    return cleaned_image, overlays, detection_results

if __name__ == "__main__":
    # Run the test
    cleaned, overlays, results = test_visual_detection()
    
    print("\n✅ Visual detection test completed!")
    print("📁 Check the 'overlay_debug' directory for step-by-step visualizations")
