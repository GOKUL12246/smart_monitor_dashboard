# heatmap.py
"""
Crowd Density Heatmap Visualization Module
Generates visual heatmaps showing crowd distribution across temple zones
Supports real-time updates and historical analysis
"""

import numpy as np
import cv2
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import json
import base64
import io

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-GUI backend for server
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib not installed - heatmap visualization limited")


# ==================== CONFIGURATION ====================
# Custom heatmap colormap (green -> yellow -> orange -> red)
HEATMAP_COLORS = [
    (0.0, '#2ECC71'),   # Green - Safe
    (0.5, '#F1C40F'),   # Yellow - Moderate  
    (0.7, '#E67E22'),   # Orange - Busy
    (0.85, '#E74C3C'),  # Red - Warning
    (1.0, '#8E44AD'),   # Purple - Critical
]

# Zone positions for temple layout (normalized 0-1)
DEFAULT_ZONE_LAYOUT = {
    "entry_gate": {"x": 0.1, "y": 0.5, "width": 0.15, "height": 0.2},
    "exit_gate": {"x": 0.9, "y": 0.5, "width": 0.15, "height": 0.2},
    "main_temple": {"x": 0.5, "y": 0.4, "width": 0.3, "height": 0.35},
    "inner_sanctum": {"x": 0.5, "y": 0.15, "width": 0.15, "height": 0.15},
    "queue_area": {"x": 0.3, "y": 0.7, "width": 0.4, "height": 0.15},
    "food_court": {"x": 0.15, "y": 0.85, "width": 0.25, "height": 0.12},
    "medical_center": {"x": 0.85, "y": 0.85, "width": 0.12, "height": 0.12},
    "parking": {"x": 0.1, "y": 0.15, "width": 0.2, "height": 0.15}
}


# ==================== HEATMAP GENERATOR CLASS ====================
class HeatmapGenerator:
    """
    Generates crowd density heatmaps for temple zones
    """
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.zone_layout = DEFAULT_ZONE_LAYOUT.copy()
        self.zone_densities = {}
        self.density_history = {}  # Historical density data
        self.lock = threading.Lock()
        
        # Create custom colormap
        self._create_colormap()
        
        # Background image (temple layout)
        self.background = None
        self._create_default_background()
    
    def _create_colormap(self):
        """Create custom colormap for heatmap"""
        if MATPLOTLIB_AVAILABLE:
            colors = [(float(pos), color) for pos, color in HEATMAP_COLORS]
            positions = [c[0] for c in colors]
            color_list = [c[1] for c in colors]
            
            # Convert hex to RGB
            rgb_colors = []
            for c in color_list:
                c = c.lstrip('#')
                rgb_colors.append(tuple(int(c[i:i+2], 16)/255 for i in (0, 2, 4)))
            
            self.cmap = LinearSegmentedColormap.from_list(
                'crowd_density', 
                list(zip(positions, rgb_colors))
            )
        else:
            self.cmap = None
    
    def _create_default_background(self):
        """Create default temple layout background"""
        # Create white background
        self.background = np.ones((self.height, self.width, 3), dtype=np.uint8) * 240
        
        # Draw grid lines
        for i in range(0, self.width, 50):
            cv2.line(self.background, (i, 0), (i, self.height), (220, 220, 220), 1)
        for i in range(0, self.height, 50):
            cv2.line(self.background, (0, i), (self.width, i), (220, 220, 220), 1)
        
        # Draw zone outlines
        for zone_id, layout in self.zone_layout.items():
            x = int(layout["x"] * self.width - layout["width"] * self.width / 2)
            y = int(layout["y"] * self.height - layout["height"] * self.height / 2)
            w = int(layout["width"] * self.width)
            h = int(layout["height"] * self.height)
            
            # Draw zone rectangle
            cv2.rectangle(self.background, (x, y), (x + w, y + h), (200, 200, 200), 2)
    
    def set_zone_layout(self, layout):
        """Set custom zone layout"""
        self.zone_layout = layout
        self._create_default_background()
    
    def update_zone_density(self, zone_id, density):
        """
        Update density for a zone
        density: 0.0 (empty) to 1.0+ (overcrowded)
        """
        with self.lock:
            self.zone_densities[zone_id] = min(density, 1.5)  # Cap at 150%
            
            # Record history
            if zone_id not in self.density_history:
                self.density_history[zone_id] = deque(maxlen=360)
            
            self.density_history[zone_id].append({
                "timestamp": datetime.now().isoformat(),
                "density": density
            })
    
    def update_from_zones(self, zones_data):
        """Update densities from zone manager data"""
        for zone in zones_data:
            zone_id = zone.get("zone_id")
            fill_pct = zone.get("fill_percentage", 0)
            density = fill_pct / 100  # Convert to 0-1 range
            self.update_zone_density(zone_id, density)
    
    def _density_to_color(self, density):
        """Convert density value to BGR color"""
        # Clamp density
        density = max(0, min(1, density))
        
        # Color gradient: Green -> Yellow -> Orange -> Red -> Purple
        if density < 0.5:
            # Green to Yellow
            r = int(46 + (241 - 46) * (density / 0.5))
            g = int(204 + (196 - 204) * (density / 0.5))
            b = int(113 + (15 - 113) * (density / 0.5))
        elif density < 0.7:
            # Yellow to Orange
            t = (density - 0.5) / 0.2
            r = int(241 + (230 - 241) * t)
            g = int(196 + (126 - 196) * t)
            b = int(15 + (34 - 15) * t)
        elif density < 0.85:
            # Orange to Red
            t = (density - 0.7) / 0.15
            r = int(230 + (231 - 230) * t)
            g = int(126 + (76 - 126) * t)
            b = int(34 + (60 - 34) * t)
        else:
            # Red to Purple
            t = (density - 0.85) / 0.15
            r = int(231 + (142 - 231) * t)
            g = int(76 + (68 - 76) * t)
            b = int(60 + (173 - 60) * t)
        
        return (b, g, r)  # BGR format for OpenCV
    
    def generate_heatmap_cv(self):
        """Generate heatmap using OpenCV (no matplotlib dependency)"""
        # Start with background
        heatmap = self.background.copy()
        
        with self.lock:
            for zone_id, layout in self.zone_layout.items():
                density = self.zone_densities.get(zone_id, 0)
                
                # Calculate zone rectangle
                x = int(layout["x"] * self.width - layout["width"] * self.width / 2)
                y = int(layout["y"] * self.height - layout["height"] * self.height / 2)
                w = int(layout["width"] * self.width)
                h = int(layout["height"] * self.height)
                
                # Get color based on density
                color = self._density_to_color(density)
                
                # Create semi-transparent overlay
                overlay = heatmap.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
                
                # Blend with original
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, heatmap, 1 - alpha, 0, heatmap)
                
                # Draw border
                border_color = (0, 0, 0) if density < 0.85 else (0, 0, 255)
                cv2.rectangle(heatmap, (x, y), (x + w, y + h), border_color, 2)
                
                # Add zone label
                label = zone_id.replace("_", " ").title()
                density_text = f"{int(density * 100)}%"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                # Zone name
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x + (w - text_w) // 2
                text_y = y + h // 2 - 5
                
                # Draw text with background
                cv2.rectangle(heatmap, (text_x - 2, text_y - text_h - 2), 
                             (text_x + text_w + 2, text_y + 2), (255, 255, 255), -1)
                cv2.putText(heatmap, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                
                # Density percentage
                (text_w2, text_h2), _ = cv2.getTextSize(density_text, font, font_scale, thickness)
                text_x2 = x + (w - text_w2) // 2
                text_y2 = text_y + text_h + 15
                
                density_color = (0, 0, 255) if density >= 0.85 else (0, 0, 0)
                cv2.rectangle(heatmap, (text_x2 - 2, text_y2 - text_h2 - 2),
                             (text_x2 + text_w2 + 2, text_y2 + 2), (255, 255, 255), -1)
                cv2.putText(heatmap, density_text, (text_x2, text_y2), font, font_scale, density_color, thickness)
        
        # Add legend
        self._add_legend(heatmap)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(heatmap, f"Updated: {timestamp}", (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return heatmap
    
    def _add_legend(self, image):
        """Add color legend to heatmap"""
        legend_x = self.width - 120
        legend_y = 20
        legend_width = 100
        legend_height = 150
        
        # Draw legend background
        cv2.rectangle(image, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (255, 255, 255), -1)
        cv2.rectangle(image, (legend_x, legend_y), 
                     (legend_x + legend_width, legend_y + legend_height), 
                     (100, 100, 100), 1)
        
        # Add title
        cv2.putText(image, "Density", (legend_x + 10, legend_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add color gradient
        bar_width = 20
        bar_height = 100
        bar_x = legend_x + 10
        bar_y = legend_y + 30
        
        for i in range(bar_height):
            density = i / bar_height
            color = self._density_to_color(density)
            cv2.rectangle(image, (bar_x, bar_y + bar_height - i - 1),
                         (bar_x + bar_width, bar_y + bar_height - i),
                         color, -1)
        
        # Add border
        cv2.rectangle(image, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 1)
        
        # Add labels
        labels = [(0, "0%"), (50, "50%"), (100, "100%")]
        for offset, label in labels:
            y_pos = bar_y + bar_height - offset
            cv2.putText(image, label, (bar_x + bar_width + 5, y_pos + 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    def generate_heatmap_matplotlib(self):
        """Generate heatmap using matplotlib (better quality)"""
        if not MATPLOTLIB_AVAILABLE:
            return self.generate_heatmap_cv()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create density grid
        grid = np.zeros((100, 100))
        
        with self.lock:
            for zone_id, layout in self.zone_layout.items():
                density = self.zone_densities.get(zone_id, 0)
                
                # Calculate zone area in grid
                x1 = int((layout["x"] - layout["width"]/2) * 100)
                x2 = int((layout["x"] + layout["width"]/2) * 100)
                y1 = int((layout["y"] - layout["height"]/2) * 100)
                y2 = int((layout["y"] + layout["height"]/2) * 100)
                
                grid[y1:y2, x1:x2] = density
        
        # Apply Gaussian blur for smooth visualization
        from scipy.ndimage import gaussian_filter
        grid_smooth = gaussian_filter(grid, sigma=3)
        
        # Plot heatmap
        im = ax.imshow(grid_smooth, cmap=self.cmap, vmin=0, vmax=1, 
                      extent=[0, 100, 100, 0], interpolation='bilinear')
        
        # Add zone labels
        for zone_id, layout in self.zone_layout.items():
            x = layout["x"] * 100
            y = layout["y"] * 100
            density = self.zone_densities.get(zone_id, 0)
            
            label = zone_id.replace("_", "\n").title()
            ax.annotate(f"{label}\n{int(density*100)}%", (x, y),
                       ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Crowd Density')
        cbar.set_ticks([0, 0.5, 0.7, 0.85, 1.0])
        cbar.set_ticklabels(['Safe', 'Moderate', 'Busy', 'Warning', 'Critical'])
        
        ax.set_title(f'Temple Crowd Heatmap - {datetime.now().strftime("%H:%M:%S")}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        # Convert to OpenCV format
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img
    
    def generate_heatmap_base64(self, use_matplotlib=False):
        """Generate heatmap and return as base64 encoded string"""
        if use_matplotlib:
            heatmap = self.generate_heatmap_matplotlib()
        else:
            heatmap = self.generate_heatmap_cv()
        
        _, buffer = cv2.imencode('.png', heatmap)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def generate_time_series_chart(self, zone_id, hours=1):
        """Generate time series chart for a zone"""
        if zone_id not in self.density_history:
            return None
        
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        history = list(self.density_history[zone_id])
        if not history:
            return None
        
        # Filter to requested time range
        cutoff = datetime.now() - timedelta(hours=hours)
        filtered = [h for h in history 
                   if datetime.fromisoformat(h["timestamp"]) > cutoff]
        
        if not filtered:
            return None
        
        timestamps = [datetime.fromisoformat(h["timestamp"]) for h in filtered]
        densities = [h["density"] * 100 for h in filtered]  # Convert to percentage
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.fill_between(timestamps, densities, alpha=0.3)
        ax.plot(timestamps, densities, linewidth=2)
        
        # Add threshold lines
        ax.axhline(y=85, color='orange', linestyle='--', label='Warning (85%)')
        ax.axhline(y=100, color='red', linestyle='--', label='Critical (100%)')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Crowd Density (%)')
        ax.set_title(f'{zone_id.replace("_", " ").title()} - Density Over Time')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 120)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def get_density_data(self):
        """Get current density data for all zones"""
        with self.lock:
            return {
                zone_id: {
                    "density": density,
                    "percentage": int(density * 100),
                    "status": self._get_status_from_density(density)
                }
                for zone_id, density in self.zone_densities.items()
            }
    
    def _get_status_from_density(self, density):
        """Get status string from density value"""
        if density >= 1.0:
            return "critical"
        elif density >= 0.85:
            return "warning"
        elif density >= 0.70:
            return "busy"
        elif density >= 0.50:
            return "moderate"
        return "safe"


# ==================== FRAME HEATMAP OVERLAY ====================
class FrameHeatmapOverlay:
    """
    Generates heatmap overlay on camera frames based on detected positions
    """
    
    def __init__(self, decay_rate=0.95, blur_size=75):
        self.decay_rate = decay_rate
        self.blur_size = blur_size
        self.heat_accumulator = None
        self.lock = threading.Lock()
    
    def update(self, frame, detections):
        """
        Update heatmap with new detections
        frame: Original camera frame
        detections: List of (x, y, w, h) bounding boxes
        """
        height, width = frame.shape[:2]
        
        with self.lock:
            # Initialize accumulator if needed
            if self.heat_accumulator is None or self.heat_accumulator.shape[:2] != (height, width):
                self.heat_accumulator = np.zeros((height, width), dtype=np.float32)
            
            # Decay existing heat
            self.heat_accumulator *= self.decay_rate
            
            # Add heat for each detection
            for (x, y, w, h) in detections:
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Add Gaussian heat at detection center
                cv2.circle(self.heat_accumulator, (center_x, center_y), 
                          max(w, h) // 2, 1.0, -1)
    
    def get_overlay(self, frame):
        """Get heatmap overlay for frame"""
        if self.heat_accumulator is None:
            return frame
        
        with self.lock:
            # Normalize and blur
            heat_normalized = cv2.normalize(self.heat_accumulator, None, 0, 255, cv2.NORM_MINMAX)
            heat_blurred = cv2.GaussianBlur(heat_normalized, (self.blur_size, self.blur_size), 0)
            
            # Convert to color
            heat_colored = cv2.applyColorMap(heat_blurred.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend with original frame
            overlay = cv2.addWeighted(frame, 0.7, heat_colored, 0.3, 0)
            
            return overlay
    
    def reset(self):
        """Reset heat accumulator"""
        with self.lock:
            if self.heat_accumulator is not None:
                self.heat_accumulator.fill(0)


# ==================== GLOBAL INSTANCES ====================
heatmap_generator = HeatmapGenerator()
frame_overlay = FrameHeatmapOverlay()


# ==================== CONVENIENCE FUNCTIONS ====================
def init_heatmap(width=800, height=600, zone_layout=None):
    """Initialize heatmap generator"""
    global heatmap_generator
    heatmap_generator = HeatmapGenerator(width, height)
    if zone_layout:
        heatmap_generator.set_zone_layout(zone_layout)
    return heatmap_generator

def update_heatmap_density(zone_id, density):
    """Update zone density"""
    heatmap_generator.update_zone_density(zone_id, density)

def update_heatmap_from_zones(zones_data):
    """Update heatmap from zone manager data"""
    heatmap_generator.update_from_zones(zones_data)

def get_heatmap_image():
    """Get heatmap as OpenCV image"""
    return heatmap_generator.generate_heatmap_cv()

def get_heatmap_base64():
    """Get heatmap as base64 string"""
    return heatmap_generator.generate_heatmap_base64()

def get_frame_with_heatmap(frame, detections):
    """Get frame with heatmap overlay"""
    frame_overlay.update(frame, detections)
    return frame_overlay.get_overlay(frame)

def get_density_data():
    """Get current density data"""
    return heatmap_generator.get_density_data()


if __name__ == "__main__":
    # Test Heatmap Generator
    print("Testing Heatmap Generator...")
    
    # Initialize
    init_heatmap(800, 600)
    
    # Simulate density updates
    test_data = [
        ("entry_gate", 0.6),
        ("exit_gate", 0.4),
        ("main_temple", 0.85),
        ("inner_sanctum", 1.1),
        ("queue_area", 0.75),
        ("food_court", 0.5),
        ("medical_center", 0.2),
        ("parking", 0.3)
    ]
    
    for zone_id, density in test_data:
        update_heatmap_density(zone_id, density)
    
    # Generate heatmap
    heatmap = get_heatmap_image()
    
    # Save to file
    cv2.imwrite("test_heatmap.png", heatmap)
    print("✅ Heatmap saved to test_heatmap.png")
    
    # Print density data
    print("\nDensity Data:")
    for zone_id, data in get_density_data().items():
        print(f"  {zone_id}: {data['percentage']}% ({data['status']})")
