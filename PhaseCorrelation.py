import cv2
import numpy as np
import time

class OptimizedPhaseCorrelation:
    def __init__(self):
        # Pre-allocate working arrays to avoid repeated allocation
        self.dft1 = None
        self.dft2 = None
        self.cross_power_spectrum = None
        
    def register_images_optimized(self, img1, img2):
        """
        Optimized phase correlation using CV_32F and in-place operations
        """
        h, w = img1.shape
        
        # Lazy allocation of working arrays
        if self.dft1 is None or self.dft1.shape[:2] != (h, w):
            self.dft1 = np.zeros((h, w, 2), dtype=np.float32)
            self.dft2 = np.zeros((h, w, 2), dtype=np.float32)
            self.cross_power_spectrum = np.zeros((h, w, 2), dtype=np.float32)
        
        # Convert to float32 in-place (critical for performance)
        img1_f32 = img1.astype(np.float32, copy=False)
        img2_f32 = img2.astype(np.float32, copy=False)
        
        # Apply window function to reduce edge effects (optional but recommended)
        # Using Hanning window for better registration accuracy
        window = self._get_hanning_window(h, w)
        img1_f32 *= window
        img2_f32 *= window
        
        # In-place DFT operations
        cv2.dft(img1_f32, self.dft1, cv2.DFT_COMPLEX_OUTPUT)
        cv2.dft(img2_f32, self.dft2, cv2.DFT_COMPLEX_OUTPUT)
        
        # Compute cross power spectrum in-place
        self._compute_cross_power_spectrum(self.dft1, self.dft2, self.cross_power_spectrum)
        
        # Inverse DFT to get correlation surface
        cv2.idft(self.cross_power_spectrum, self.cross_power_spectrum, 
                cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        
        # Find peak location (sub-pixel accuracy)
        correlation_surface = self.cross_power_spectrum[:, :, 0]
        shift = self._find_subpixel_peak(correlation_surface)
        
        return shift
    
    def _compute_cross_power_spectrum(self, dft1, dft2, output):
        """
        Compute cross power spectrum: F1 * conj(F2) / |F1 * conj(F2)|
        All operations done in-place for memory efficiency
        """
        # Complex multiplication: F1 * conj(F2)
        # Real part: a1*a2 + b1*b2
        # Imag part: b1*a2 - a1*b2
        output[:, :, 0] = dft1[:, :, 0] * dft2[:, :, 0] + dft1[:, :, 1] * dft2[:, :, 1]
        output[:, :, 1] = dft1[:, :, 1] * dft2[:, :, 0] - dft1[:, :, 0] * dft2[:, :, 1]
        
        # Compute magnitude for normalization
        magnitude = np.sqrt(output[:, :, 0]**2 + output[:, :, 1]**2)
        
        # Avoid division by zero
        magnitude[magnitude < 1e-10] = 1e-10
        
        # Normalize to get cross power spectrum
        output[:, :, 0] /= magnitude
        output[:, :, 1] /= magnitude
    
    def _get_hanning_window(self, h, w):
        """
        Create 2D Hanning window for edge effect reduction
        """
        if not hasattr(self, '_window_cache') or self._window_cache.shape != (h, w):
            hann_h = np.hanning(h).reshape(-1, 1)
            hann_w = np.hanning(w).reshape(1, -1)
            self._window_cache = (hann_h * hann_w).astype(np.float32)
        return self._window_cache
    
    def _find_subpixel_peak(self, correlation_surface):
        """
        Find sub-pixel peak location using parabolic interpolation
        """
        # Find integer peak
        max_loc = np.unravel_index(np.argmax(correlation_surface), correlation_surface.shape)
        y_max, x_max = max_loc
        
        h, w = correlation_surface.shape
        
        # Sub-pixel refinement using parabolic interpolation
        if 1 <= y_max < h-1 and 1 <= x_max < w-1:
            # Get neighborhood values
            c = correlation_surface[y_max, x_max]
            
            # X direction
            l = correlation_surface[y_max, x_max-1]
            r = correlation_surface[y_max, x_max+1]
            dx = 0.5 * (r - l) / (2*c - l - r) if (2*c - l - r) != 0 else 0
            
            # Y direction  
            u = correlation_surface[y_max-1, x_max]
            d = correlation_surface[y_max+1, x_max]
            dy = 0.5 * (d - u) / (2*c - u - d) if (2*c - u - d) != 0 else 0
            
            # Sub-pixel coordinates
            x_shift = x_max + dx
            y_shift = y_max + dy
        else:
            x_shift, y_shift = x_max, y_max
        
        # Convert to shift relative to center
        x_shift = x_shift if x_shift <= w//2 else x_shift - w
        y_shift = y_shift if y_shift <= h//2 else y_shift - h
        
        return (x_shift, y_shift)

# Performance comparison and usage example
def benchmark_registration_methods():
    """
    Compare optimized vs standard registration methods
    """
    # Create test images (simulating 1MB TIFF)
    img1 = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
    
    # Create shifted version
    shift_x, shift_y = 5.3, -2.7  # Sub-pixel shift
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img2 = cv2.warpAffine(img1, M, (1024, 1024))
    
    # Add some noise
    noise = np.random.normal(0, 10, img2.shape).astype(np.float32)
    img2 = np.clip(img2.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Test optimized method
    optimizer = OptimizedPhaseCorrelation()
    
    # Warm-up run
    _ = optimizer.register_images_optimized(img1, img2)
    
    # Benchmark
    start_time = time.time()
    num_runs = 20
    
    for _ in range(num_runs):
        detected_shift = optimizer.register_images_optimized(img1, img2)
    
    avg_time = (time.time() - start_time) / num_runs * 1000  # Convert to ms
    
    print(f"Average registration time: {avg_time:.1f} ms")
    print(f"True shift: ({shift_x:.1f}, {shift_y:.1f})")
    print(f"Detected shift: ({detected_shift[0]:.1f}, {detected_shift[1]:.1f})")
    print(f"Error: ({abs(detected_shift[0] - shift_x):.2f}, {abs(detected_shift[1] - shift_y):.2f}) pixels")

# Memory usage optimization for batch processing
class BatchImageProcessor:
    def __init__(self, max_workers=4):
        self.correlator = OptimizedPhaseCorrelation()
        self.max_workers = max_workers
        
    def process_image_batch(self, image_paths, reference_image_path):
        """
        Process batch of images with memory optimization
        """
        import tifffile
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Load reference image once
        ref_image = tifffile.imread(reference_image_path)
        
        results = []
        
        # Process in chunks to control memory usage
        chunk_size = 50  # Adjust based on available memory
        
        for i in range(0, len(image_paths), chunk_size):
            chunk = image_paths[i:i + chunk_size]
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_image, path, ref_image): path 
                    for path in chunk
                }
                
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        result = future.result()
                        results.append((path, result))
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
                        results.append((path, None))
        
        return results
    
    def _process_single_image(self, image_path, reference_image):
        """
        Process single image - called in worker process
        """
        import tifffile
        
        # Load image
        image = tifffile.imread(image_path)
        
        # Perform registration
        correlator = OptimizedPhaseCorrelation()
        shift = correlator.register_images_optimized(reference_image, image)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(image)
        
        return {
            'shift': shift,
            'quality': quality_metrics,
            'processing_time_ms': 0  # Would be measured in actual implementation
        }
    
    def _calculate_quality_metrics(self, image):
        """
        Fast quality metric calculation using optimized OpenCV functions
        """
        # Convert to float32 for SIMD optimization
        img_f32 = image.astype(np.float32)
        
        # Sharpness using Laplacian variance (very fast)
        laplacian = cv2.Laplacian(img_f32, cv2.CV_32F)
        sharpness = laplacian.var()
        
        # Contrast using RMS
        contrast = img_f32.std()
        
        # SNR estimation using local standard deviation
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(img_f32, -1, kernel)
        noise_std = (img_f32 - local_mean).std()
        snr_db = 20 * np.log10(img_f32.mean() / noise_std) if noise_std > 0 else float('inf')
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'snr_db': float(snr_db)
        }

if __name__ == "__main__":
    # Run benchmark
    print("Running registration benchmark...")
    benchmark_registration_methods()
    
    # Example usage for production
    print("\nExample batch processing setup:")
    processor = BatchImageProcessor(max_workers=4)
    print("Batch processor initialized with optimized registration and quality metrics")
