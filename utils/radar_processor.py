import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
from scipy import ndimage
from skimage import feature, measure, morphology, filters, segmentation

class RadarProcessor:
    """Class for processing and analyzing radar data to detect tornadic signatures."""
    
    def __init__(self):
        """Initialize the radar processor."""
        self.reflectivity_threshold = 40  # dBZ threshold for significant precipitation 
        self.velocity_threshold = 15      # m/s threshold for strong rotation
        self.hook_reflectivity_threshold = 35  # dBZ threshold for hook echo detection
        
    def process_radar_data(self, radar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process radar data and detect weather features.
        
        Args:
            radar_data: Dictionary containing radar data arrays
            
        Returns:
            Dictionary containing processed results with detected features
        """
        try:
            # Extract data arrays
            reflectivity = np.array(radar_data.get('reflectivity', []))
            velocity = np.array(radar_data.get('velocity', []))
            spectrum_width = np.array(radar_data.get('spectrum_width', []))
            
            # Check for empty arrays
            if reflectivity.size == 0 or velocity.size == 0:
                return self._default_empty_results()
            
            # Detect mesocyclones from velocity data
            mesocyclone = self.detect_mesocyclones(velocity, spectrum_width)
            
            # Detect hook echoes from reflectivity data
            hook_echo = self.detect_hook_echoes(reflectivity)
            
            # Calculate storm motion
            storm_motion = self.calculate_storm_motion(radar_data.get('timestamps', []), reflectivity)
            
            # Calculate rotation metrics
            rotation_metrics = self.calculate_rotation_metrics(velocity)
            
            return {
                'mesocyclone': mesocyclone,
                'hook_echo': hook_echo,
                'storm_motion': storm_motion,
                'rotation_metrics': rotation_metrics
            }
            
        except Exception as e:
            print(f"Error processing radar data: {str(e)}")
            return self._default_empty_results()
    
    def _default_empty_results(self) -> Dict[str, Any]:
        """Return default empty results when processing fails."""
        return {
            'mesocyclone': {'detected': False, 'strength': 0.0, 'location': (0, 0), 'diameter': 0, 'depth': 0},
            'hook_echo': {'detected': False, 'confidence': 0.0, 'location': (0, 0)},
            'storm_motion': (0, 0),
            'rotation_metrics': {'max_rotation': 0, 'mean_rotation': 0, 'rotation_area': 0}
        }
    
    def detect_mesocyclones(self, velocity: np.ndarray, spectrum_width: np.ndarray = None) -> Dict[str, Any]:
        """Detect mesocyclones in radar velocity data.
        
        A mesocyclone is characterized by a velocity couplet (adjacent regions of 
        winds moving in opposite directions) in radar velocity data.
        
        Args:
            velocity: Doppler velocity data array (m/s)
            spectrum_width: Optional spectrum width data for additional filtering
            
        Returns:
            Dictionary with mesocyclone detection results
        """
        # Ensure we have a 2D array (use the latest scan if it's 3D)
        if len(velocity.shape) > 2:
            velocity_scan = velocity[-1]  # Use the latest scan
        else:
            velocity_scan = velocity
            
        # Apply Gaussian smoothing to reduce noise
        smoothed = filters.gaussian(velocity_scan, sigma=1.0)
        
        # Calculate the gradient of the velocity field
        # This highlights areas where velocity changes rapidly (potential rotation)
        gradient_y, gradient_x = np.gradient(smoothed)
        rotation = gradient_x - gradient_y  # Simplified rotation calculation
        
        # Find regions of strong rotation
        rotation_threshold = 0.2
        rotation_mask = np.abs(rotation) > rotation_threshold
        
        # Apply morphological operations to clean up the mask
        cleaned_mask = morphology.remove_small_objects(rotation_mask, min_size=5)
        
        # Label connected regions
        labeled_mask, num_features = ndimage.label(cleaned_mask)
        
        if num_features == 0:
            return {'detected': False, 'strength': 0.0, 'location': (0, 0), 'diameter': 0, 'depth': 0}
        
        # Find the strongest rotation region
        rotation_strengths = []
        locations = []
        diameters = []
        
        for i in range(1, num_features + 1):
            region = labeled_mask == i
            region_rotation = np.abs(rotation[region]).mean()
            rotation_strengths.append(region_rotation)
            
            # Calculate centroid and size
            props = measure.regionprops(region.astype(int))[0]
            y, x = props.centroid
            area = props.area
            diameter = 2 * np.sqrt(area / np.pi)  # Estimate diameter
            
            locations.append((y, x))
            diameters.append(diameter)
        
        # Find the strongest mesocyclone
        if rotation_strengths:
            max_idx = np.argmax(rotation_strengths)
            max_strength = rotation_strengths[max_idx]
            max_location = locations[max_idx]
            max_diameter = diameters[max_idx]
            
            # Normalize strength to 0-1 range
            normalized_strength = min(max_strength / 0.5, 1.0)
            
            return {
                'detected': normalized_strength > 0.5,
                'strength': float(normalized_strength),
                'location': (float(max_location[1]), float(max_location[0])),  # Convert to lon, lat format
                'diameter': float(max_diameter * 100),  # Scale to realistic diameter in meters
                'depth': float(3000 + 2000 * normalized_strength)  # Estimate depth based on strength
            }
        else:
            return {'detected': False, 'strength': 0.0, 'location': (0, 0), 'diameter': 0, 'depth': 0}
    
    def detect_hook_echoes(self, reflectivity: np.ndarray) -> Dict[str, Any]:
        """Detect hook echoes in reflectivity data.
        
        Hook echoes are curved appendages on the right rear of a supercell,
        often indicating the presence of a mesocyclone and potentially a tornado.
        
        Args:
            reflectivity: Radar reflectivity data array (dBZ)
            
        Returns:
            Dictionary with hook echo detection results
        """
        # Ensure we have a 2D array (use the latest scan if it's 3D)
        if len(reflectivity.shape) > 2:
            reflectivity_scan = reflectivity[-1]  # Use the latest scan
        else:
            reflectivity_scan = reflectivity
        
        # Threshold reflectivity to identify significant precipitation
        binary = reflectivity_scan > self.hook_reflectivity_threshold
        
        # Clean up the binary image
        binary = morphology.remove_small_objects(binary, min_size=10)
        binary = morphology.binary_closing(binary, morphology.disk(2))
        
        # Label connected storm cells
        labeled, num_cells = ndimage.label(binary)
        
        if num_cells == 0:
            return {'detected': False, 'confidence': 0.0, 'location': (0, 0)}
        
        # Analyze shape of each cell to look for hook-like formations
        hook_scores = []
        hook_locations = []
        
        for i in range(1, num_cells + 1):
            cell = labeled == i
            
            # Skip small cells
            if np.sum(cell) < 20:
                continue
                
            # Get cell contour
            contours = measure.find_contours(cell.astype(float), 0.5)
            
            if not contours:
                continue
                
            # Get the longest contour
            longest_contour = max(contours, key=len)
            
            # Calculate curvature along the contour
            dx = np.gradient(longest_contour[:, 1])
            dy = np.gradient(longest_contour[:, 0])
            
            # Second derivatives
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            # Curvature calculation
            curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
            
            # Look for high curvature regions (potential hooks)
            high_curvature = curvature > np.percentile(curvature, 90)
            curvature_regions, num_regions = ndimage.label(high_curvature)
            
            # Calculate cell convexity - hook echoes typically create concavities
            hull = morphology.convex_hull_image(cell)
            convexity_defects = hull & ~cell
            convexity_score = np.sum(convexity_defects) / np.sum(cell)
            
            # Calculate hook score based on curvature and convexity
            hook_score = convexity_score * np.mean(curvature[high_curvature]) if np.any(high_curvature) else 0
            
            # Find centroid of the cell
            props = measure.regionprops(cell.astype(int))[0]
            y, x = props.centroid
            
            hook_scores.append(hook_score)
            hook_locations.append((y, x))
        
        # Find the best hook echo candidate
        if hook_scores:
            max_idx = np.argmax(hook_scores)
            max_score = hook_scores[max_idx]
            max_location = hook_locations[max_idx]
            
            # Normalize score to 0-1 range
            confidence = min(max_score / 0.3, 1.0)
            
            return {
                'detected': confidence > 0.6,
                'confidence': float(confidence),
                'location': (float(max_location[1]), float(max_location[0]))  # Convert to lon, lat format
            }
        else:
            return {'detected': False, 'confidence': 0.0, 'location': (0, 0)}
    
    def calculate_rotation_metrics(self, velocity: np.ndarray) -> Dict[str, float]:
        """Calculate rotation metrics from velocity data.
        
        Args:
            velocity: Doppler velocity data
            
        Returns:
            Dictionary of rotation metrics
        """
        # Ensure we have a 2D array
        if len(velocity.shape) > 2:
            velocity_scan = velocity[-1]  # Use the latest scan
        else:
            velocity_scan = velocity
            
        # Calculate rotation as curl of velocity field
        gradient_y, gradient_x = np.gradient(velocity_scan)
        rotation = gradient_x - gradient_y
        
        # Calculate metrics
        max_rotation = float(np.max(np.abs(rotation)))
        mean_rotation = float(np.mean(np.abs(rotation)))
        
        # Calculate area of significant rotation
        rotation_area = float(np.sum(np.abs(rotation) > 0.2))
        
        return {
            'max_rotation': max_rotation,
            'mean_rotation': mean_rotation,
            'rotation_area': rotation_area
        }
        
    def calculate_storm_motion(self, timestamps: List[datetime], reflectivity: np.ndarray) -> Tuple[float, float]:
        """Calculate storm motion vector using cross-correlation between consecutive frames.
        
        Args:
            timestamps: List of timestamp objects for each radar scan
            reflectivity: 3D array of reflectivity data with time as first dimension
            
        Returns:
            Tuple of (u, v) components of storm motion in m/s
        """
        # Check if we have enough data for motion calculation
        if len(reflectivity.shape) < 3 or reflectivity.shape[0] < 2:
            return (0.0, 0.0)
            
        # Take two consecutive frames
        frame1 = reflectivity[-2]
        frame2 = reflectivity[-1]
        
        # Use OpenCV's phase correlation method to find the shift
        try:
            # Ensure frames are properly formatted for OpenCV
            frame1_norm = cv2.normalize(frame1, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            frame2_norm = cv2.normalize(frame2, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            
            # Calculate phase correlation
            shift, response = cv2.phaseCorrelate(frame1_norm, frame2_norm)
            
            # Convert shift to velocity
            # Assuming radar images are ~100km across and timestamps are in order
            dx, dy = shift
            
            # Calculate time difference in seconds (default to 5 minutes if timestamps unavailable)
            if len(timestamps) >= 2:
                dt = (timestamps[-1] - timestamps[-2]).total_seconds()
            else:
                dt = 300  # Default to 5 minutes
                
            # Scale factors (pixel to meters)
            scale = 100000 / max(frame1.shape)  # Assuming 100km across
            
            # Calculate velocity components
            u = dx * scale / dt
            v = dy * scale / dt
            
            return (float(u), float(v))
            
        except Exception as e:
            print(f"Error calculating storm motion: {str(e)}")
            return (0.0, 0.0) 