# zone_manager.py
"""
Multi-Zone Detection System for Temple Crowd Monitoring
Manages multiple camera zones with individual crowd limits and status tracking
"""

import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json

# ==================== CONFIGURATION ====================
class ZoneStatus(Enum):
    SAFE = "safe"              # Under 50% capacity
    MODERATE = "moderate"      # 50-70% capacity
    BUSY = "busy"              # 70-85% capacity
    WARNING = "warning"        # 85-100% capacity
    CRITICAL = "critical"      # Over 100% capacity
    EMERGENCY = "emergency"    # Critical + additional risk factors
    CLOSED = "closed"          # Zone temporarily closed

class ZoneType(Enum):
    ENTRY_GATE = "entry_gate"
    EXIT_GATE = "exit_gate"
    MAIN_TEMPLE = "main_temple"
    INNER_SANCTUM = "inner_sanctum"
    QUEUE_AREA = "queue_area"
    FOOD_COURT = "food_court"
    REST_AREA = "rest_area"
    MEDICAL_CENTER = "medical_center"
    PARKING = "parking"
    CORRIDOR = "corridor"
    CUSTOM = "custom"

# Default zone configurations for temples
DEFAULT_ZONE_CONFIGS = {
    "entry_gate": {
        "name": "Entry Gate",
        "type": ZoneType.ENTRY_GATE,
        "capacity": 3,  # TESTING: Set low for easy email alert testing (change to 30 for production)
        "warning_threshold": 0.70,
        "critical_threshold": 0.85,
        "camera_id": 0,
        "priority": 1,
        "flow_direction": "inward"
    },
    "exit_gate": {
        "name": "Exit Gate",
        "type": ZoneType.EXIT_GATE,
        "capacity": 30,  # Customized: Increased for realistic gate capacity
        "warning_threshold": 0.70,
        "critical_threshold": 0.85,
        "camera_id": 1,
        "priority": 1,
        "flow_direction": "outward"
    },
    "main_temple": {
        "name": "Main Temple Hall",
        "type": ZoneType.MAIN_TEMPLE,
        "capacity": 100,  # Customized: Typical temple hall capacity
        "warning_threshold": 0.75,
        "critical_threshold": 0.90,
        "camera_id": 2,
        "priority": 2,
        "flow_direction": "circular"
    },
    "inner_sanctum": {
        "name": "Inner Sanctum",
        "type": ZoneType.INNER_SANCTUM,
        "capacity": 20,  # Customized: Small sacred area capacity
        "warning_threshold": 0.60,
        "critical_threshold": 0.80,
        "camera_id": 3,
        "priority": 3,
        "flow_direction": "linear"
    },
    "queue_area": {
        "name": "Queue Area",
        "type": ZoneType.QUEUE_AREA,
        "capacity": 200,  # Customized: Large queue capacity
        "warning_threshold": 0.85,
        "critical_threshold": 0.95,
        "camera_id": 4,
        "priority": 2,
        "flow_direction": "linear"
    },
    "food_court": {
        "name": "Food Court / Prasad Area",
        "type": ZoneType.FOOD_COURT,
        "capacity": 120,  # Customized: Medium dining area capacity
        "warning_threshold": 0.80,
        "critical_threshold": 0.95,
        "camera_id": 5,
        "priority": 1,
        "flow_direction": "random"
    },
    "medical_center": {
        "name": "Medical Center",
        "type": ZoneType.MEDICAL_CENTER,
        "capacity": 15,  # Customized: Small medical facility capacity
        "warning_threshold": 0.70,
        "critical_threshold": 0.90,
        "camera_id": 6,
        "priority": 3,
        "flow_direction": "static"
    },
    "parking": {
        "name": "Parking Area",
        "type": ZoneType.PARKING,
        "capacity": 300,  # Customized: Medium parking lot capacity
        "warning_threshold": 0.85,
        "critical_threshold": 0.95,
        "camera_id": 7,
        "priority": 1,
        "flow_direction": "random"
    }
}


# ==================== ZONE CLASS ====================
class Zone:
    """Individual zone representation with crowd tracking"""
    
    def __init__(self, zone_id, name, zone_type, capacity, 
                 warning_threshold=0.85, critical_threshold=1.0,
                 camera_id=None, priority=1, flow_direction="random"):
        self.zone_id = zone_id
        self.name = name
        self.zone_type = zone_type if isinstance(zone_type, ZoneType) else ZoneType(zone_type)
        self.capacity = capacity
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.camera_id = camera_id
        self.priority = priority
        self.flow_direction = flow_direction
        
        # Current state
        self.current_count = 0
        self.status = ZoneStatus.SAFE
        self.is_closed = False
        self.last_updated = datetime.now()
        
        # Historical data
        self.count_history = deque(maxlen=360)  # 1 hour at 10-sec intervals
        self.hourly_averages = deque(maxlen=24)  # 24 hours
        self.peak_count_today = 0
        self.total_count_today = 0
        self.status_changes = deque(maxlen=50)
        
        # Alerts
        self.alert_cooldown = 60  # seconds between alerts
        self.last_alert_time = None
        self.consecutive_warnings = 0
        
        # Fall detection tracking
        self.fall_detected = False
        self.fall_detection_time = None
        self.fall_location = None
    
    def update_count(self, count):
        """Update current crowd count"""
        self.current_count = count
        self.last_updated = datetime.now()
        
        # Record history
        self.count_history.append({
            "timestamp": self.last_updated.isoformat(),
            "count": count
        })
        
        # Update peak
        if count > self.peak_count_today:
            self.peak_count_today = count
        
        self.total_count_today += count
        
        # Update status
        old_status = self.status
        self._update_status()
        
        # Track status changes
        if old_status != self.status:
            self.status_changes.append({
                "timestamp": self.last_updated.isoformat(),
                "from": old_status.value,
                "to": self.status.value
            })
        
        return self.status
    
    def _update_status(self):
        """Update zone status based on current count"""
        if self.is_closed:
            self.status = ZoneStatus.CLOSED
            return
        
        if self.fall_detected:
            self.status = ZoneStatus.EMERGENCY
            return
        
        fill_percentage = self.get_fill_percentage() / 100
        
        if fill_percentage >= self.critical_threshold:
            self.status = ZoneStatus.CRITICAL
            self.consecutive_warnings += 1
        elif fill_percentage >= self.warning_threshold:
            self.status = ZoneStatus.WARNING
            self.consecutive_warnings += 1
        elif fill_percentage >= 0.70:
            self.status = ZoneStatus.BUSY
            self.consecutive_warnings = 0
        elif fill_percentage >= 0.50:
            self.status = ZoneStatus.MODERATE
            self.consecutive_warnings = 0
        else:
            self.status = ZoneStatus.SAFE
            self.consecutive_warnings = 0
    
    def get_fill_percentage(self):
        """Get current fill percentage"""
        if self.capacity <= 0:
            return 0
        return (self.current_count / self.capacity) * 100
    
    def get_available_capacity(self):
        """Get remaining capacity"""
        return max(0, self.capacity - self.current_count)
    
    def should_alert(self):
        """Check if alert should be triggered"""
        if self.status in [ZoneStatus.WARNING, ZoneStatus.CRITICAL, ZoneStatus.EMERGENCY]:
            if self.last_alert_time is None:
                return True
            elapsed = (datetime.now() - self.last_alert_time).total_seconds()
            return elapsed >= self.alert_cooldown
        return False
    
    def mark_alert_sent(self):
        """Mark that an alert was sent"""
        self.last_alert_time = datetime.now()
    
    def set_fall_detected(self, detected, location=None):
        """Update fall detection state"""
        self.fall_detected = detected
        if detected:
            self.fall_detection_time = datetime.now()
            self.fall_location = location
        else:
            self.fall_detection_time = None
            self.fall_location = None
        self._update_status()
    
    def close_zone(self):
        """Close the zone"""
        self.is_closed = True
        self._update_status()
    
    def open_zone(self):
        """Open the zone"""
        self.is_closed = False
        self._update_status()
    
    def get_average_count(self, minutes=30):
        """Get average count over past N minutes"""
        if not self.count_history:
            return 0
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_counts = [
            h["count"] for h in self.count_history
            if datetime.fromisoformat(h["timestamp"]) > cutoff
        ]
        
        return sum(recent_counts) / len(recent_counts) if recent_counts else 0
    
    def get_trend(self):
        """Get crowd trend (increasing, decreasing, stable)"""
        if len(self.count_history) < 6:
            return "stable"
        
        recent = list(self.count_history)[-6:]
        first_half = sum(h["count"] for h in recent[:3]) / 3
        second_half = sum(h["count"] for h in recent[3:]) / 3
        
        diff_percentage = ((second_half - first_half) / max(first_half, 1)) * 100
        
        if diff_percentage > 10:
            return "increasing"
        elif diff_percentage < -10:
            return "decreasing"
        return "stable"
    
    def to_dict(self):
        """Convert zone to dictionary"""
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "type": self.zone_type.value,
            "capacity": self.capacity,
            "current_count": self.current_count,
            "fill_percentage": round(self.get_fill_percentage(), 1),
            "available_capacity": self.get_available_capacity(),
            "status": self.status.value,
            "is_closed": self.is_closed,
            "priority": self.priority,
            "last_updated": self.last_updated.isoformat(),
            "peak_count_today": self.peak_count_today,
            "average_count": round(self.get_average_count(), 1),
            "trend": self.get_trend(),
            "fall_detected": self.fall_detected,
            "consecutive_warnings": self.consecutive_warnings
        }


# ==================== ZONE MANAGER CLASS ====================
class ZoneManager:
    """
    Manages multiple zones in a temple complex
    Tracks crowd flow, triggers alerts, and provides analytics
    """
    
    def __init__(self):
        self.zones = {}
        self.lock = threading.Lock()
        self.total_visitors_today = 0
        self.peak_visitors_today = 0
        self.start_time = datetime.now()
        self.alert_callbacks = []
        self.global_alert_cooldown = 30  # seconds
        self.last_global_alert = None
        
        # Total capacity
        self.total_capacity = 0
        
        # Alert history
        self.alert_history = deque(maxlen=100)
    
    def add_zone(self, zone_id, **kwargs):
        """Add a new zone"""
        with self.lock:
            zone = Zone(zone_id, **kwargs)
            self.zones[zone_id] = zone
            self.total_capacity += zone.capacity
            print(f"✅ Zone added: {zone.name} (capacity: {zone.capacity})")
            return zone
    
    def add_default_zones(self, zone_ids=None):
        """Add default temple zones"""
        if zone_ids is None:
            zone_ids = list(DEFAULT_ZONE_CONFIGS.keys())
        
        for zone_id in zone_ids:
            if zone_id in DEFAULT_ZONE_CONFIGS:
                config = DEFAULT_ZONE_CONFIGS[zone_id].copy()
                self.add_zone(
                    zone_id=zone_id,
                    name=config.pop("name"),
                    zone_type=config.pop("type"),
                    **config
                )
    
    def remove_zone(self, zone_id):
        """Remove a zone"""
        with self.lock:
            if zone_id in self.zones:
                self.total_capacity -= self.zones[zone_id].capacity
                del self.zones[zone_id]
                print(f"🗑️ Zone removed: {zone_id}")
    
    def get_zone(self, zone_id):
        """Get zone by ID"""
        return self.zones.get(zone_id)
    
    def update_zone_count(self, zone_id, count):
        """Update crowd count for a zone"""
        zone = self.zones.get(zone_id)
        if zone:
            old_status = zone.status
            new_status = zone.update_count(count)
            
            # Check for status change alerts
            if old_status != new_status:
                self._handle_status_change(zone, old_status, new_status)
            
            # Update global stats
            self._update_global_stats()
            
            return new_status
        return None
    
    def _handle_status_change(self, zone, old_status, new_status):
        """Handle zone status changes"""
        # Record in history
        self.alert_history.append({
            "timestamp": datetime.now().isoformat(),
            "zone_id": zone.zone_id,
            "zone_name": zone.name,
            "from_status": old_status.value,
            "to_status": new_status.value,
            "count": zone.current_count,
            "capacity": zone.capacity
        })
        
        # Trigger callbacks for significant changes
        if new_status in [ZoneStatus.WARNING, ZoneStatus.CRITICAL, ZoneStatus.EMERGENCY]:
            self._trigger_alert_callbacks(zone, new_status)
    
    def _trigger_alert_callbacks(self, zone, status):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(zone, status)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def register_alert_callback(self, callback):
        """Register callback for zone alerts"""
        self.alert_callbacks.append(callback)
    
    def _update_global_stats(self):
        """Update global visitor statistics"""
        total = sum(z.current_count for z in self.zones.values())
        if total > self.peak_visitors_today:
            self.peak_visitors_today = total
    
    def get_total_count(self):
        """Get total people across all zones"""
        return sum(z.current_count for z in self.zones.values())
    
    def get_total_capacity(self):
        """Get total capacity across all zones"""
        return self.total_capacity
    
    def get_overall_fill_percentage(self):
        """Get overall fill percentage"""
        if self.total_capacity <= 0:
            return 0
        return (self.get_total_count() / self.total_capacity) * 100
    
    def get_zones_by_status(self, status):
        """Get zones with specific status"""
        return [z for z in self.zones.values() if z.status == status]
    
    def get_critical_zones(self):
        """Get zones in critical or emergency status"""
        return [z for z in self.zones.values() 
                if z.status in [ZoneStatus.CRITICAL, ZoneStatus.EMERGENCY]]
    
    def get_warning_zones(self):
        """Get zones in warning or worse status"""
        return [z for z in self.zones.values() 
                if z.status in [ZoneStatus.WARNING, ZoneStatus.CRITICAL, ZoneStatus.EMERGENCY]]
    
    def get_all_zones_summary(self):
        """Get summary of all zones"""
        return [zone.to_dict() for zone in self.zones.values()]
    
    def get_dashboard_data(self):
        """Get data formatted for dashboard display"""
        zones_data = self.get_all_zones_summary()
        
        # Count zones by status
        status_counts = {}
        for status in ZoneStatus:
            status_counts[status.value] = len([z for z in zones_data if z["status"] == status.value])
        
        return {
            "total_count": self.get_total_count(),
            "total_capacity": self.total_capacity,
            "overall_fill_percentage": round(self.get_overall_fill_percentage(), 1),
            "peak_visitors_today": self.peak_visitors_today,
            "zones": zones_data,
            "status_counts": status_counts,
            "critical_zones": len(self.get_critical_zones()),
            "warning_zones": len(self.get_warning_zones()),
            "uptime_minutes": int((datetime.now() - self.start_time).total_seconds() / 60),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_zone_recommendations(self):
        """Get recommendations for crowd management"""
        recommendations = []
        
        for zone in self.zones.values():
            fill_pct = zone.get_fill_percentage()
            trend = zone.get_trend()
            
            if zone.status == ZoneStatus.CRITICAL:
                recommendations.append({
                    "zone": zone.name,
                    "priority": "critical",
                    "action": "Immediately redirect visitors to alternate routes",
                    "details": f"Zone at {fill_pct:.0f}% capacity, trend: {trend}"
                })
            elif zone.status == ZoneStatus.WARNING and trend == "increasing":
                recommendations.append({
                    "zone": zone.name,
                    "priority": "high",
                    "action": "Start crowd control measures, prepare alternate routes",
                    "details": f"Zone at {fill_pct:.0f}% and increasing"
                })
            elif zone.status == ZoneStatus.BUSY and trend == "increasing":
                recommendations.append({
                    "zone": zone.name,
                    "priority": "medium",
                    "action": "Monitor closely, consider flow control",
                    "details": f"Zone at {fill_pct:.0f}% with increasing trend"
                })
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))
        
        return recommendations
    
    def set_fall_detected(self, zone_id, detected, location=None):
        """Set fall detection for a zone"""
        zone = self.zones.get(zone_id)
        if zone:
            zone.set_fall_detected(detected, location)
            if detected:
                self._trigger_alert_callbacks(zone, ZoneStatus.EMERGENCY)
    
    def close_zone(self, zone_id):
        """Close a zone"""
        zone = self.zones.get(zone_id)
        if zone:
            zone.close_zone()
    
    def open_zone(self, zone_id):
        """Open a zone"""
        zone = self.zones.get(zone_id)
        if zone:
            zone.open_zone()
    
    def reset_daily_stats(self):
        """Reset daily statistics (call at midnight)"""
        for zone in self.zones.values():
            zone.peak_count_today = 0
            zone.total_count_today = 0
        self.peak_visitors_today = 0
        self.total_visitors_today = 0
        print("📊 Daily statistics reset")
    
    def export_data(self, filepath=None):
        """Export zone data to JSON"""
        data = {
            "export_time": datetime.now().isoformat(),
            "zones": self.get_all_zones_summary(),
            "alert_history": list(self.alert_history),
            "global_stats": {
                "total_capacity": self.total_capacity,
                "peak_visitors_today": self.peak_visitors_today,
                "uptime_minutes": int((datetime.now() - self.start_time).total_seconds() / 60)
            }
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"📁 Data exported to {filepath}")
        
        return data


# ==================== GLOBAL INSTANCE ====================
zone_manager = ZoneManager()


# ==================== CONVENIENCE FUNCTIONS ====================
def init_temple_zones(custom_zones=None):
    """Initialize temple zones with default or custom configuration"""
    global zone_manager
    
    # Clear existing zones instead of creating new instance
    zone_manager.zones.clear()
    zone_manager.total_capacity = 0
    
    if custom_zones:
        for zone_config in custom_zones:
            zone_manager.add_zone(**zone_config)
    else:
        zone_manager.add_default_zones()
    
    print(f"✅ Initialized {len(zone_manager.zones)} zones")
    return zone_manager

def update_zone_count(zone_id, count):
    """Update crowd count for a zone"""
    return zone_manager.update_zone_count(zone_id, count)

def get_zone_status(zone_id):
    """Get status of a specific zone"""
    zone = zone_manager.get_zone(zone_id)
    return zone.to_dict() if zone else None

def get_all_zones():
    """Get all zones summary"""
    return zone_manager.get_all_zones_summary()

def get_dashboard_data():
    """Get dashboard data"""
    return zone_manager.get_dashboard_data()

def get_critical_zones():
    """Get critical zones"""
    return [z.to_dict() for z in zone_manager.get_critical_zones()]

def set_fall_detected(zone_id, detected, location=None):
    """Set fall detection for a zone"""
    zone_manager.set_fall_detected(zone_id, detected, location)

def close_zone(zone_id):
    """Close a zone"""
    zone_manager.close_zone(zone_id)

def open_zone(zone_id):
    """Open a zone"""
    zone_manager.open_zone(zone_id)

def update_zone_capacity(zone_id, capacity, warning_threshold=None, critical_threshold=None):
    """Update zone capacity and thresholds"""
    zone = zone_manager.get_zone(zone_id)
    if zone:
        zone.capacity = int(capacity)
        if warning_threshold is not None:
            zone.warning_threshold = float(warning_threshold)
        if critical_threshold is not None:
            zone.critical_threshold = float(critical_threshold)
        print(f"✅ Updated {zone.name}: capacity={capacity}, warning={warning_threshold}, critical={critical_threshold}")
        return True
    return False

def reset_zones_to_defaults():
    """Reset all zones to default capacity values"""
    global zone_manager
    # Re-initialize with defaults
    init_temple_zones()
    print("✅ All zones reset to default capacities")
    return True


if __name__ == "__main__":
    # Test Zone Manager
    print("Testing Zone Manager...")
    
    # Initialize with default zones
    init_temple_zones()
    
    # Simulate crowd updates
    update_zone_count("entry_gate", 50)
    update_zone_count("main_temple", 150)
    update_zone_count("food_court", 70)
    update_zone_count("inner_sanctum", 28)
    
    # Get dashboard data
    dashboard = get_dashboard_data()
    print(f"\nTotal Visitors: {dashboard['total_count']}")
    print(f"Overall Fill: {dashboard['overall_fill_percentage']}%")
    print(f"Critical Zones: {dashboard['critical_zones']}")
    
    print("\nZone Status:")
    for zone in dashboard['zones']:
        print(f"  {zone['name']}: {zone['current_count']}/{zone['capacity']} ({zone['status']})")
    
    # Test critical threshold
    print("\n--- Simulating overcrowding ---")
    update_zone_count("inner_sanctum", 35)  # Over capacity
    
    critical = get_critical_zones()
    print(f"Critical zones: {[z['name'] for z in critical]}")
    
    # Get recommendations
    print("\nRecommendations:")
    for rec in zone_manager.get_zone_recommendations():
        print(f"  [{rec['priority'].upper()}] {rec['zone']}: {rec['action']}")
