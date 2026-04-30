"""
RF Communication Module for Temple Monitoring System
Provides radio/LoRa communication for offline alerts
"""

import serial
import serial.tools.list_ports
import threading
import queue
import time
from datetime import datetime
from enum import Enum

class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class RecipientGroup(Enum):
    SECURITY = "security"
    MEDICAL = "medical"
    MANAGEMENT = "management"
    TEMPLE_ADMIN = "admin"
    ALL = "all"

class RFModule:
    """RF/LoRa module handler"""
    
    def __init__(self, port=None, baudrate=9600, protocol="LoRa"):
        self.port = port
        self.baudrate = baudrate
        self.protocol = protocol
        self.serial_conn = None
        self.connected = False
        self.running = False
        self.receive_thread = None
        self.message_queue = queue.Queue()
    
    def auto_detect_port(self):
        """Auto-detect RF module port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'USB' in port.description or 'Serial' in port.description:
                return port.device
        return None
    
    def connect(self):
        """Initialize connection to RF module"""
        try:
            if self.port is None:
                self.port = self.auto_detect_port()
            
            if self.port:
                self.serial_conn = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=0.5,
                    write_timeout=0.5
                )
                time.sleep(0.2)
                self.connected = True
                print(f"✅ RF Module connected on {self.port} ({self.protocol})")
                return True
            else:
                print(f"⚠️ No RF module detected - running in simulation mode")
                self.connected = False
                return False
        except KeyboardInterrupt:
            print(f"⚠️ RF connection interrupted - skipping")
            self.connected = False
            return False
        except Exception as e:
            print(f"❌ RF connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close RF connection"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.connected = False
    
    def send(self, message, recipient="ALL"):
        """Send message via RF"""
        if not self.connected:
            return False
        
        try:
            data = f"{recipient}:{message}\n".encode('utf-8')
            self.serial_conn.write(data)
            return True
        except Exception as e:
            print(f"❌ RF send failed: {e}")
            return False
    
    def start(self):
        """Start receiving messages"""
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
    
    def stop(self):
        """Stop receiving messages"""
        self.running = False
    
    def _receive_loop(self):
        """Background thread to receive messages"""
        while self.running and self.connected:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        self.message_queue.put(line)
            except Exception as e:
                if self.running:
                    print(f"❌ RF receive error: {e}")
            time.sleep(0.1)

class AlertBroadcaster:
    """Manages alert broadcasting via RF"""
    
    def __init__(self):
        self.rf = RFModule()
        self.pending_alerts = []
        self.acknowledged = set()
    
    def initialize(self):
        """Initialize RF communication"""
        if self.rf.connect():
            self.rf.start()
            return True
        return False
    
    def shutdown(self):
        """Shutdown RF communication"""
        self.rf.stop()
        self.rf.disconnect()
    
    def broadcast(self, message, priority=AlertPriority.MEDIUM, recipient=RecipientGroup.ALL):
        """Broadcast alert message"""
        if self.rf.connected:
            self.rf.send(message, recipient.value)
        # Store for tracking
        alert_id = f"{int(time.time())}_{len(self.pending_alerts)}"
        self.pending_alerts.append({
            'id': alert_id,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now(),
            'acknowledged': False
        })
        return alert_id
    
    def broadcast_escalation(self, zone_name, level):
        """Broadcast escalation alert"""
        message = f"ESCALATION Level {level}: {zone_name} requires management attention"
        return self.broadcast(message, AlertPriority.HIGH, RecipientGroup.MANAGEMENT)
    
    def get_pending(self):
        """Get pending alerts"""
        return [a for a in self.pending_alerts if not a['acknowledged']]
    
    def acknowledge_alert(self, alert_id):
        """Mark alert as acknowledged"""
        for alert in self.pending_alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                self.acknowledged.add(alert_id)
                return True
        return False

# Global alert broadcaster instance
alert_broadcaster = AlertBroadcaster()

# Public API functions
def init_rf_communication():
    """Initialize RF communication system"""
    return alert_broadcaster.initialize()

def shutdown_rf_communication():
    """Shutdown RF communication system"""
    alert_broadcaster.shutdown()

def broadcast_crowd_alert(zone_name, count, limit):
    """Broadcast crowd alert"""
    message = f"CROWD ALERT: {zone_name} - {count}/{limit} people"
    return alert_broadcaster.broadcast(message, AlertPriority.HIGH, RecipientGroup.SECURITY)

def broadcast_fall_alert(zone_name, details=""):
    """Broadcast fall detection alert"""
    message = f"FALL DETECTED: {zone_name}"
    if details:
        message += f" - {details}"
    return alert_broadcaster.broadcast(message, AlertPriority.CRITICAL, RecipientGroup.MEDICAL)

def broadcast_emergency(message):
    """Broadcast emergency message"""
    return alert_broadcaster.broadcast(message, AlertPriority.CRITICAL, RecipientGroup.ALL)

def broadcast_escalation(zone_name, level):
    """Broadcast escalation alert"""
    message = f"ESCALATION Level {level}: {zone_name} requires management attention"
    return alert_broadcaster.broadcast(message, AlertPriority.HIGH, RecipientGroup.MANAGEMENT)

def get_pending_alerts():
    """Get list of pending alerts"""
    return alert_broadcaster.get_pending()

def acknowledge(alert_id):
    """Acknowledge an alert"""
    return alert_broadcaster.acknowledge_alert(alert_id)
