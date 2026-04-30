# environmental_risk.py
"""
Environmental Risk Engine
========================
Simulates sensor inputs and calculates risk scores for temple safety
Features:
- Temperature monitoring
- Air Quality Index (AQI) monitoring
- Rain detection
- Crowd density integration
- Risk scoring algorithm (0-100)
- Risk level classification
"""

import random
import threading
import time
from datetime import datetime
from collections import deque
import json

class EnvironmentalRiskEngine:
    def __init__(self, update_interval=30):
        """
        Initialize Environmental Risk Engine
        
        Args:
            update_interval: Seconds between sensor updates (default: 30)
        """
        self.update_interval = update_interval
        
        # Sensor data
        self.temperature = 28.0  # °C
        self.aqi = 50  # Air Quality Index
        self.rain = 0  # 0 or 1
        self.crowd_density = 0  # People count from YOLO
        self.max_capacity = 500  # Maximum safe capacity
        
        # Risk calculation weights
        self.weights = {
            "crowd_density": 0.40,  # 40%
            "temperature": 0.25,    # 25%
            "aqi": 0.20,            # 20%
            "rain": 0.15            # 15%
        }
        
        # Risk history
        self.risk_history = deque(maxlen=100)  # Store last 100 readings
        
        # Simulation parameters
        self.simulate_sensors = True
        self.temp_range = (25, 42)  # Temperature range (°C)
        self.aqi_range = (0, 300)   # AQI range
        self.rain_probability = 0.1  # 10% chance of rain
        
        # Background thread
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Risk thresholds
        self.risk_levels = {
            "LOW": (0, 30),
            "MEDIUM": (30, 60),
            "HIGH": (60, 80),
            "CRITICAL": (80, 100)
        }
        
        print("✅ Environmental Risk Engine initialized")
        print(f"   - Update interval: {update_interval}s")
        print(f"   - Weights: Crowd={self.weights['crowd_density']*100}%, "
              f"Temp={self.weights['temperature']*100}%, "
              f"AQI={self.weights['aqi']*100}%, "
              f"Rain={self.weights['rain']*100}%")
    
    def start(self):
        """Start the risk engine and sensor simulation"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print("🔄 Environmental Risk Engine started")
    
    def stop(self):
        """Stop the risk engine"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("⏹️ Environmental Risk Engine stopped")
    
    def _update_loop(self):
        """Background loop to update sensor data"""
        while self.running:
            if self.simulate_sensors:
                self._simulate_sensors()
            
            # Calculate risk
            risk_data = self.calculate_risk()
            
            # Store in history
            with self.lock:
                self.risk_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "temperature": self.temperature,
                    "aqi": self.aqi,
                    "rain": self.rain,
                    "crowd_density": self.crowd_density,
                    "risk_score": risk_data["risk_score"],
                    "risk_level": risk_data["risk_level"]
                })
            
            time.sleep(self.update_interval)
    
    def _simulate_sensors(self):
        """Simulate sensor readings with realistic variations"""
        with self.lock:
            # Temperature: Gradual changes with random walk
            temp_change = random.uniform(-0.5, 0.5)
            self.temperature = max(self.temp_range[0], 
                                  min(self.temp_range[1], 
                                      self.temperature + temp_change))
            
            # AQI: Random walk with occasional spikes
            if random.random() < 0.05:  # 5% chance of spike
                self.aqi = random.randint(150, 250)
            else:
                aqi_change = random.randint(-10, 10)
                self.aqi = max(self.aqi_range[0], 
                              min(self.aqi_range[1], 
                                  self.aqi + aqi_change))
            
            # Rain: Random on/off
            if random.random() < self.rain_probability:
                self.rain = 1
            else:
                self.rain = 0
    
    def update_crowd_density(self, count):
        """Update crowd density from detection system"""
        with self.lock:
            self.crowd_density = count
    
    def set_max_capacity(self, capacity):
        """Set maximum safe capacity"""
        with self.lock:
            self.max_capacity = capacity
    
    def set_sensor_data(self, temperature=None, aqi=None, rain=None):
        """Manually set sensor data (disables simulation for those sensors)"""
        with self.lock:
            if temperature is not None:
                self.temperature = temperature
            if aqi is not None:
                self.aqi = aqi
            if rain is not None:
                self.rain = rain
    
    def calculate_risk(self):
        """
        Calculate environmental risk score (0-100)
        
        Returns:
            dict: Risk assessment with score, level, and factors
        """
        with self.lock:
            # 1. Crowd Density Score (0-100)
            crowd_percentage = (self.crowd_density / max(self.max_capacity, 1)) * 100
            crowd_score = min(100, crowd_percentage)  # Cap at 100
            
            # 2. Temperature Score (0-100)
            # Optimal: 20-30°C, Risk increases above 30°C
            if self.temperature <= 30:
                temp_score = 0
            elif self.temperature <= 35:
                temp_score = (self.temperature - 30) * 20  # 0-100
            else:
                temp_score = min(100, 100 + (self.temperature - 35) * 10)
            
            # 3. AQI Score (0-100)
            # Good: 0-50, Moderate: 51-100, Unhealthy: 101+
            if self.aqi <= 50:
                aqi_score = 0
            elif self.aqi <= 100:
                aqi_score = (self.aqi - 50) * 0.5  # 0-25
            elif self.aqi <= 150:
                aqi_score = 25 + (self.aqi - 100) * 0.5  # 25-50
            else:
                aqi_score = min(100, 50 + (self.aqi - 150) * 0.33)  # 50-100
            
            # 4. Rain Score (0-100)
            rain_score = 100 if self.rain == 1 else 0
            
            # Combined Risk Score (weighted average)
            risk_score = (
                crowd_score * self.weights["crowd_density"] +
                temp_score * self.weights["temperature"] +
                aqi_score * self.weights["aqi"] +
                rain_score * self.weights["rain"]
            )
            
            # Apply special risk rules
            risk_modifiers = []
            
            # Rule 1: High temp + high density = dehydration risk
            if self.temperature > 35 and crowd_percentage > 70:
                risk_score = min(100, risk_score * 1.2)
                risk_modifiers.append("Dehydration risk: High temp + density")
            
            # Rule 2: High AQI = respiratory risk
            if self.aqi > 150:
                risk_score = min(100, risk_score * 1.15)
                risk_modifiers.append("Respiratory risk: Poor air quality")
            
            # Rule 3: Rain + high density = slip/stampede risk
            if self.rain == 1 and crowd_percentage > 60:
                risk_score = min(100, risk_score * 1.3)
                risk_modifiers.append("Slip/stampede risk: Rain + density")
            
            # Determine risk level
            risk_level = self._get_risk_level(risk_score)
            
            return {
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "factors": {
                    "crowd_density": {
                        "value": self.crowd_density,
                        "percentage": round(crowd_percentage, 1),
                        "score": round(crowd_score, 2),
                        "weight": self.weights["crowd_density"]
                    },
                    "temperature": {
                        "value": round(self.temperature, 1),
                        "unit": "°C",
                        "score": round(temp_score, 2),
                        "weight": self.weights["temperature"]
                    },
                    "aqi": {
                        "value": int(self.aqi),
                        "score": round(aqi_score, 2),
                        "weight": self.weights["aqi"]
                    },
                    "rain": {
                        "value": self.rain,
                        "active": self.rain == 1,
                        "score": round(rain_score, 2),
                        "weight": self.weights["rain"]
                    }
                },
                "modifiers": risk_modifiers,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_risk_level(self, risk_score):
        """Determine risk level from score"""
        for level, (min_score, max_score) in self.risk_levels.items():
            if min_score <= risk_score < max_score:
                return level
        return "CRITICAL"  # Default to critical if >= 80
    
    def get_current_status(self):
        """Get current environmental status"""
        risk_data = self.calculate_risk()
        
        with self.lock:
            return {
                "sensors": {
                    "temperature": round(self.temperature, 1),
                    "aqi": int(self.aqi),
                    "rain": self.rain,
                    "crowd_density": self.crowd_density
                },
                "risk": risk_data,
                "capacity": {
                    "current": self.crowd_density,
                    "maximum": self.max_capacity,
                    "percentage": round((self.crowd_density / max(self.max_capacity, 1)) * 100, 1)
                }
            }
    
    def get_risk_history(self, limit=50):
        """Get recent risk history"""
        with self.lock:
            history = list(self.risk_history)
        return history[-limit:]
    
    def get_statistics(self):
        """Get statistical summary of risk data"""
        with self.lock:
            history = list(self.risk_history)
        
        if not history:
            return {
                "available": False,
                "message": "No historical data available"
            }
        
        risk_scores = [item["risk_score"] for item in history]
        
        return {
            "available": True,
            "count": len(history),
            "risk_score": {
                "current": risk_scores[-1] if risk_scores else 0,
                "average": round(sum(risk_scores) / len(risk_scores), 2),
                "min": round(min(risk_scores), 2),
                "max": round(max(risk_scores), 2)
            },
            "alerts": {
                "critical": sum(1 for item in history if item["risk_level"] == "CRITICAL"),
                "high": sum(1 for item in history if item["risk_level"] == "HIGH"),
                "medium": sum(1 for item in history if item["risk_level"] == "MEDIUM"),
                "low": sum(1 for item in history if item["risk_level"] == "LOW")
            }
        }
    
    def export_data(self, filepath="environmental_risk_log.json"):
        """Export risk history to JSON file"""
        try:
            with self.lock:
                data = {
                    "export_time": datetime.now().isoformat(),
                    "configuration": {
                        "weights": self.weights,
                        "max_capacity": self.max_capacity,
                        "risk_levels": self.risk_levels
                    },
                    "history": list(self.risk_history)
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"📊 Exported {len(self.risk_history)} risk records to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return False

# Global instance
risk_engine = None

def init_risk_engine(update_interval=30):
    """Initialize the environmental risk engine"""
    global risk_engine
    if risk_engine is None:
        risk_engine = EnvironmentalRiskEngine(update_interval)
        risk_engine.start()
    return risk_engine

def get_risk_engine():
    """Get the risk engine instance"""
    global risk_engine
    if risk_engine is None:
        init_risk_engine()
    return risk_engine

def shutdown_risk_engine():
    """Shutdown the risk engine"""
    global risk_engine
    if risk_engine:
        risk_engine.stop()
        risk_engine = None
