# escalation.py
"""
Automatic Alert Escalation System for Temple Crowd Monitoring
Automatically escalates alerts if not acknowledged within time limits
Supports priority-based escalation chains and notification routing
"""

import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json

# ==================== CONFIGURATION ====================
class EscalationLevel(Enum):
    LEVEL_1 = 1  # Initial alert - volunteers only
    LEVEL_2 = 2  # First escalation - add security/police
    LEVEL_3 = 3  # Second escalation - add temple admin
    LEVEL_4 = 4  # Third escalation - add medical/emergency
    LEVEL_5 = 5  # Final escalation - all responders + external help

class AlertType(Enum):
    CROWD_WARNING = "crowd_warning"
    CROWD_EXCEEDED = "crowd_exceeded"
    FALL_DETECTED = "fall_detected"
    ZONE_CLOSURE = "zone_closure"
    EVACUATION = "evacuation"
    STAMPEDE_RISK = "stampede_risk"
    MEDICAL_EMERGENCY = "medical_emergency"
    FIRE = "fire"
    CUSTOM = "custom"

# Escalation timing (seconds before next level)
ESCALATION_TIMINGS = {
    AlertType.CROWD_WARNING: [120, 180, 300, 600],      # 2, 3, 5, 10 minutes
    AlertType.CROWD_EXCEEDED: [60, 120, 180, 300],      # 1, 2, 3, 5 minutes
    AlertType.FALL_DETECTED: [30, 60, 90, 120],         # 30s, 1, 1.5, 2 minutes
    AlertType.ZONE_CLOSURE: [60, 120, 180, 300],
    AlertType.EVACUATION: [30, 60, 90, 120],
    AlertType.STAMPEDE_RISK: [15, 30, 45, 60],          # Very fast escalation
    AlertType.MEDICAL_EMERGENCY: [30, 60, 90, 120],
    AlertType.FIRE: [15, 30, 45, 60],                   # Very fast escalation
    AlertType.CUSTOM: [120, 180, 300, 600]              # Default timing
}

# Responders by escalation level
ESCALATION_RESPONDERS = {
    EscalationLevel.LEVEL_1: ["volunteers"],
    EscalationLevel.LEVEL_2: ["volunteers", "security", "police"],
    EscalationLevel.LEVEL_3: ["volunteers", "security", "police", "temple_admin"],
    EscalationLevel.LEVEL_4: ["volunteers", "security", "police", "temple_admin", "medical"],
    EscalationLevel.LEVEL_5: ["volunteers", "security", "police", "temple_admin", "medical", "fire_dept", "district_admin"]
}

# Alert priority scores (higher = more urgent)
ALERT_PRIORITIES = {
    AlertType.FIRE: 100,
    AlertType.STAMPEDE_RISK: 95,
    AlertType.EVACUATION: 90,
    AlertType.MEDICAL_EMERGENCY: 85,
    AlertType.FALL_DETECTED: 80,
    AlertType.CROWD_EXCEEDED: 70,
    AlertType.ZONE_CLOSURE: 50,
    AlertType.CROWD_WARNING: 40,
    AlertType.CUSTOM: 30
}


# ==================== ALERT CLASS ====================
class Alert:
    """Individual alert with escalation tracking"""
    
    def __init__(self, alert_id, alert_type, zone, message, 
                 data=None, priority_override=None):
        self.alert_id = alert_id
        self.alert_type = alert_type if isinstance(alert_type, AlertType) else AlertType(alert_type)
        self.zone = zone
        self.message = message
        self.data = data or {}
        
        self.created_at = datetime.now()
        self.last_escalated_at = self.created_at
        self.current_level = EscalationLevel.LEVEL_1
        
        # Priority (can be overridden)
        self.priority = priority_override or ALERT_PRIORITIES.get(self.alert_type, 30)
        
        # Status tracking
        self.acknowledged = False
        self.acknowledged_by = None
        self.acknowledged_at = None
        self.resolved = False
        self.resolved_at = None
        self.resolved_by = None
        
        # Escalation history
        self.escalation_history = [{
            "level": 1,
            "timestamp": self.created_at.isoformat(),
            "responders": ESCALATION_RESPONDERS[EscalationLevel.LEVEL_1]
        }]
        
        # Notification tracking
        self.notifications_sent = []
    
    def get_time_since_creation(self):
        """Get seconds since alert creation"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def get_time_since_last_escalation(self):
        """Get seconds since last escalation"""
        return (datetime.now() - self.last_escalated_at).total_seconds()
    
    def should_escalate(self):
        """Check if alert should be escalated"""
        if self.acknowledged or self.resolved:
            return False
        
        if self.current_level.value >= EscalationLevel.LEVEL_5.value:
            return False
        
        timings = ESCALATION_TIMINGS.get(self.alert_type, ESCALATION_TIMINGS[AlertType.CUSTOM])
        current_index = self.current_level.value - 1
        
        if current_index >= len(timings):
            return False
        
        time_limit = timings[current_index]
        return self.get_time_since_last_escalation() >= time_limit
    
    def escalate(self):
        """Escalate to next level"""
        if self.current_level.value >= EscalationLevel.LEVEL_5.value:
            return False
        
        new_level = EscalationLevel(self.current_level.value + 1)
        self.current_level = new_level
        self.last_escalated_at = datetime.now()
        
        # Increase priority with each escalation
        self.priority = min(100, self.priority + 10)
        
        self.escalation_history.append({
            "level": new_level.value,
            "timestamp": self.last_escalated_at.isoformat(),
            "responders": ESCALATION_RESPONDERS[new_level]
        })
        
        return True
    
    def acknowledge(self, responder_id, notes=""):
        """Acknowledge the alert"""
        self.acknowledged = True
        self.acknowledged_by = responder_id
        self.acknowledged_at = datetime.now()
        
        self.escalation_history.append({
            "action": "acknowledged",
            "timestamp": self.acknowledged_at.isoformat(),
            "responder": responder_id,
            "notes": notes
        })
        
        return True
    
    def resolve(self, responder_id, resolution_notes=""):
        """Mark alert as resolved"""
        self.resolved = True
        self.resolved_by = responder_id
        self.resolved_at = datetime.now()
        
        self.escalation_history.append({
            "action": "resolved",
            "timestamp": self.resolved_at.isoformat(),
            "responder": responder_id,
            "notes": resolution_notes
        })
        
        return True
    
    def get_current_responders(self):
        """Get list of responders for current level"""
        return ESCALATION_RESPONDERS.get(self.current_level, [])
    
    def to_dict(self):
        """Convert alert to dictionary"""
        return {
            "alert_id": self.alert_id,
            "type": self.alert_type.value,
            "zone": self.zone,
            "message": self.message,
            "data": self.data,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "current_level": self.current_level.value,
            "level_name": f"Level {self.current_level.value}",
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "responders": self.get_current_responders(),
            "time_elapsed_seconds": int(self.get_time_since_creation()),
            "history": self.escalation_history
        }


# ==================== ESCALATION MANAGER CLASS ====================
class EscalationManager:
    """
    Manages alert escalation across the temple monitoring system
    """
    
    def __init__(self):
        self.active_alerts = {}  # alert_id -> Alert
        self.resolved_alerts = deque(maxlen=200)  # Historical resolved alerts
        self.alert_counter = 0
        self.lock = threading.Lock()
        
        # Callbacks for notifications
        self.escalation_callbacks = []
        self.acknowledgment_callbacks = []
        self.resolution_callbacks = []
        
        # Background escalation checker
        self.running = False
        self.check_thread = None
        self.check_interval = 10  # seconds
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "total_escalations": 0,
            "total_acknowledged": 0,
            "total_resolved": 0,
            "average_response_time": 0
        }
    
    def start(self):
        """Start escalation monitoring"""
        self.running = True
        self.check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self.check_thread.start()
        print("⏰ Escalation Manager started")
    
    def stop(self):
        """Stop escalation monitoring"""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=2)
        print("⏰ Escalation Manager stopped")
    
    def _check_loop(self):
        """Background loop to check for escalations"""
        while self.running:
            try:
                self._process_escalations()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Escalation check error: {e}")
    
    def _process_escalations(self):
        """Process pending escalations"""
        with self.lock:
            for alert_id, alert in list(self.active_alerts.items()):
                if alert.should_escalate():
                    if alert.escalate():
                        print(f"⬆️ Alert {alert_id} escalated to Level {alert.current_level.value}")
                        self.stats["total_escalations"] += 1
                        self._trigger_escalation_callbacks(alert)
    
    def _generate_alert_id(self):
        """Generate unique alert ID"""
        self.alert_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ALT-{timestamp}-{self.alert_counter:04d}"
    
    def create_alert(self, alert_type, zone, message, data=None, priority_override=None):
        """Create a new alert"""
        with self.lock:
            alert_id = self._generate_alert_id()
            
            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                zone=zone,
                message=message,
                data=data,
                priority_override=priority_override
            )
            
            self.active_alerts[alert_id] = alert
            self.stats["total_alerts"] += 1
            
            print(f"🚨 Alert created: {alert_id} - {alert_type} in {zone}")
            
            # Trigger callbacks
            self._trigger_escalation_callbacks(alert)
            
            return alert
    
    def acknowledge_alert(self, alert_id, responder_id, notes=""):
        """Acknowledge an alert"""
        with self.lock:
            alert = self.active_alerts.get(alert_id)
            if alert:
                alert.acknowledge(responder_id, notes)
                self.stats["total_acknowledged"] += 1
                
                # Calculate response time
                response_time = alert.get_time_since_creation()
                self._update_average_response_time(response_time)
                
                print(f"✅ Alert {alert_id} acknowledged by {responder_id}")
                
                # Trigger callbacks
                for callback in self.acknowledgment_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"Acknowledgment callback error: {e}")
                
                return True
        return False
    
    def resolve_alert(self, alert_id, responder_id, notes=""):
        """Resolve an alert"""
        with self.lock:
            alert = self.active_alerts.get(alert_id)
            if alert:
                alert.resolve(responder_id, notes)
                self.stats["total_resolved"] += 1
                
                # Move to resolved
                self.resolved_alerts.append(alert)
                del self.active_alerts[alert_id]
                
                print(f"✅ Alert {alert_id} resolved by {responder_id}")
                
                # Trigger callbacks
                for callback in self.resolution_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        print(f"Resolution callback error: {e}")
                
                return True
        return False
    
    def _update_average_response_time(self, new_time):
        """Update average response time"""
        total = self.stats["total_acknowledged"]
        current_avg = self.stats["average_response_time"]
        self.stats["average_response_time"] = ((current_avg * (total - 1)) + new_time) / total
    
    def _trigger_escalation_callbacks(self, alert):
        """Trigger escalation notification callbacks"""
        for callback in self.escalation_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Escalation callback error: {e}")
    
    def register_escalation_callback(self, callback):
        """Register callback for escalation events"""
        self.escalation_callbacks.append(callback)
    
    def register_acknowledgment_callback(self, callback):
        """Register callback for acknowledgment events"""
        self.acknowledgment_callbacks.append(callback)
    
    def register_resolution_callback(self, callback):
        """Register callback for resolution events"""
        self.resolution_callbacks.append(callback)
    
    def get_active_alerts(self, zone=None, alert_type=None, min_level=None):
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        if zone:
            alerts = [a for a in alerts if a.zone == zone]
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if min_level:
            alerts = [a for a in alerts if a.current_level.value >= min_level]
        
        # Sort by priority (highest first)
        alerts.sort(key=lambda a: (a.priority, a.current_level.value), reverse=True)
        
        return [a.to_dict() for a in alerts]
    
    def get_alert(self, alert_id):
        """Get specific alert"""
        alert = self.active_alerts.get(alert_id)
        return alert.to_dict() if alert else None
    
    def get_unacknowledged_alerts(self):
        """Get unacknowledged active alerts"""
        with self.lock:
            alerts = [a for a in self.active_alerts.values() if not a.acknowledged]
            alerts.sort(key=lambda a: a.priority, reverse=True)
            return [a.to_dict() for a in alerts]
    
    def get_alerts_by_level(self, level):
        """Get alerts at specific escalation level"""
        with self.lock:
            level_enum = EscalationLevel(level) if isinstance(level, int) else level
            alerts = [a for a in self.active_alerts.values() if a.current_level == level_enum]
            return [a.to_dict() for a in alerts]
    
    def get_critical_alerts(self):
        """Get alerts at level 3 or higher"""
        with self.lock:
            alerts = [a for a in self.active_alerts.values() if a.current_level.value >= 3]
            alerts.sort(key=lambda a: a.priority, reverse=True)
            return [a.to_dict() for a in alerts]
    
    def get_statistics(self):
        """Get escalation statistics"""
        with self.lock:
            return {
                **self.stats,
                "active_alerts": len(self.active_alerts),
                "unacknowledged": len([a for a in self.active_alerts.values() if not a.acknowledged]),
                "by_level": {
                    f"level_{i}": len([a for a in self.active_alerts.values() 
                                      if a.current_level.value == i])
                    for i in range(1, 6)
                }
            }
    
    def get_dashboard_data(self):
        """Get data for dashboard display"""
        with self.lock:
            active = self.get_active_alerts()
            
            # Group by zone
            by_zone = {}
            for alert in active:
                zone = alert["zone"]
                if zone not in by_zone:
                    by_zone[zone] = []
                by_zone[zone].append(alert)
            
            # Recent escalations
            recent_escalations = [
                a for a in active 
                if a["current_level"] > 1
            ][:5]
            
            return {
                "total_active": len(active),
                "unacknowledged": len([a for a in active if not a["acknowledged"]]),
                "critical": len([a for a in active if a["current_level"] >= 3]),
                "alerts_by_zone": by_zone,
                "recent_escalations": recent_escalations,
                "highest_level": max([a["current_level"] for a in active], default=0),
                "statistics": self.get_statistics()
            }
    
    def clear_resolved_history(self):
        """Clear resolved alerts history"""
        self.resolved_alerts.clear()
        print("🗑️ Resolved alerts history cleared")


# ==================== GLOBAL INSTANCE ====================
escalation_manager = EscalationManager()


# ==================== CONVENIENCE FUNCTIONS ====================
def init_escalation_system():
    """Initialize escalation system"""
    escalation_manager.start()
    return escalation_manager

def shutdown_escalation_system():
    """Shutdown escalation system"""
    escalation_manager.stop()

def create_crowd_warning_alert(zone, count, limit):
    """Create crowd warning alert"""
    percentage = int((count / limit) * 100) if limit > 0 else 0
    message = f"Crowd at {percentage}% capacity ({count}/{limit} people)"
    return escalation_manager.create_alert(
        AlertType.CROWD_WARNING,
        zone,
        message,
        data={"count": count, "limit": limit, "percentage": percentage}
    )

def create_crowd_exceeded_alert(zone, count, limit):
    """Create crowd exceeded alert"""
    excess = count - limit
    message = f"OVERCROWDED! {count} people detected (limit: {limit}, excess: {excess})"
    return escalation_manager.create_alert(
        AlertType.CROWD_EXCEEDED,
        zone,
        message,
        data={"count": count, "limit": limit, "excess": excess}
    )

def create_fall_alert(zone, location_details=""):
    """Create fall detection alert"""
    message = f"Person fallen! {location_details}"
    return escalation_manager.create_alert(
        AlertType.FALL_DETECTED,
        zone,
        message,
        data={"location_details": location_details}
    )

def create_evacuation_alert(zone, reason):
    """Create evacuation alert"""
    message = f"EVACUATE IMMEDIATELY! {reason}"
    return escalation_manager.create_alert(
        AlertType.EVACUATION,
        zone,
        message,
        data={"reason": reason},
        priority_override=95
    )

def create_stampede_risk_alert(zone, count, trend):
    """Create stampede risk alert"""
    message = f"STAMPEDE RISK! High density ({count}) with {trend} trend"
    return escalation_manager.create_alert(
        AlertType.STAMPEDE_RISK,
        zone,
        message,
        data={"count": count, "trend": trend},
        priority_override=95
    )

def acknowledge(alert_id, responder_id, notes=""):
    """Acknowledge an alert"""
    return escalation_manager.acknowledge_alert(alert_id, responder_id, notes)

def resolve(alert_id, responder_id, notes=""):
    """Resolve an alert"""
    return escalation_manager.resolve_alert(alert_id, responder_id, notes)

def get_active_alerts(zone=None):
    """Get active alerts"""
    return escalation_manager.get_active_alerts(zone=zone)

def get_unacknowledged():
    """Get unacknowledged alerts"""
    return escalation_manager.get_unacknowledged_alerts()

def get_critical():
    """Get critical alerts"""
    return escalation_manager.get_critical_alerts()

def get_statistics():
    """Get escalation statistics"""
    return escalation_manager.get_statistics()


if __name__ == "__main__":
    # Test Escalation System
    print("Testing Escalation System...")
    
    # Initialize
    init_escalation_system()
    
    # Create some test alerts
    alert1 = create_crowd_warning_alert("Main Temple", 85, 100)
    time.sleep(1)
    
    alert2 = create_crowd_exceeded_alert("Inner Sanctum", 40, 30)
    time.sleep(1)
    
    alert3 = create_fall_alert("Food Court", "Near counter 3")
    
    # Print active alerts
    print("\nActive Alerts:")
    for alert in get_active_alerts():
        print(f"  {alert['alert_id']}: {alert['type']} in {alert['zone']} (Level {alert['current_level']})")
    
    # Simulate acknowledgment
    print("\nAcknowledging alert...")
    acknowledge(alert1.alert_id, "VOL001", "On my way")
    
    # Wait for escalation
    print("\nWaiting for escalation (30 seconds)...")
    time.sleep(35)
    
    # Check escalation
    print("\nAfter escalation:")
    for alert in get_active_alerts():
        print(f"  {alert['alert_id']}: Level {alert['current_level']}, Acknowledged: {alert['acknowledged']}")
    
    # Get statistics
    print("\nStatistics:")
    stats = get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    shutdown_escalation_system()
