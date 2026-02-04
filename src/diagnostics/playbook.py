"""
Fault playbook with recommended technician actions
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


# Fault playbook - mapping fault types to recommended actions
FAULT_PLAYBOOK = {
    "normal": {
        "description": "Normal operation - no faults detected",
        "severity": "none",
        "actions": [
            "Continue normal monitoring"
        ]
    },
    "condenser_fouling": {
        "description": "Condenser heat exchanger fouling reducing heat rejection",
        "severity": "medium",
        "actions": [
            "Inspect condenser coil for dirt/debris",
            "Clean condenser coil (chemical wash if needed)",
            "Verify condenser fan operation and airflow",
            "Check condenser approach temperature / pressure trends",
            "Confirm head pressure stability after cleaning"
        ]
    },
    "evaporator_fouling": {
        "description": "Evaporator coil fouling reducing heat absorption",
        "severity": "medium",
        "actions": [
            "Inspect evaporator coil for dust/ice buildup",
            "Check air filter condition and replace if clogged",
            "Clean evaporator coil",
            "Verify airflow across evaporator (fan, dampers)",
            "Confirm suction pressure/temperature stabilize after cleaning"
        ]
    },
    "liquid_line_restriction": {
        "description": "Restriction in liquid line causing abnormal pressure/flow behavior",
        "severity": "high",
        "actions": [
            "Inspect liquid line filter drier for restriction",
            "Check for kinked/blocked liquid line sections",
            "Measure pressure drop across filter drier/valves",
            "Check sight glass (if available) for flashing/bubbles",
            "Replace restricted components and evacuate/recharge properly"
        ]
    },
    "refrigerant_overcharge": {
        "description": "Excess refrigerant charge leading to elevated head pressure",
        "severity": "high",
        "actions": [
            "Verify sensor calibration for condensing pressure",
            "Check condenser airflow (fan, coil cleanliness)",
            "Recover refrigerant to manufacturer spec charge",
            "Confirm superheat/subcooling within expected range",
            "Monitor condensing pressure stability after correction"
        ]
    },
    "refrigerant_undercharge": {
        "description": "Insufficient refrigerant charge reducing cooling performance",
        "severity": "high",
        "actions": [
            "Check for refrigerant leaks (joints, coils, valves)",
            "Repair leaks and perform proper evacuation",
            "Recharge to manufacturer spec charge",
            "Verify superheat/subcooling and suction pressure behavior",
            "Monitor for reoccurrence indicating active leak"
        ]
    },
    "supply_fan_failure": {
        "description": "Supply fan is not operating or operating at reduced capacity",
        "severity": "high",
        "actions": [
            "Check supply fan motor for power",
            "Verify fan belt tension and condition",
            "Inspect air filter for blockage",
            "Check for mechanical obstructions",
            "Measure fan motor current draw",
            "Test motor capacitor if present",
            "Replace fan motor if defective"
        ]
    },
    "return_fan_failure": {
        "description": "Return fan is not operating or operating at reduced capacity",
        "severity": "high",
        "actions": [
            "Check return fan motor for power",
            "Verify fan belt tension and condition",
            "Inspect return air filter for blockage",
            "Check for mechanical obstructions",
            "Measure fan motor current draw",
            "Test motor capacitor if present",
            "Replace fan motor if defective"
        ]
    },
    "compressor_failure": {
        "description": "Compressor is not operating or operating at reduced capacity",
        "severity": "high",
        "actions": [
            "Check compressor contactor for power",
            "Verify compressor motor current draw",
            "Check refrigerant charge level",
            "Inspect compressor inlet/outlet pressures",
            "Listen for unusual compressor noise",
            "Check compressor oil level",
            "Replace compressor if defective"
        ]
    },
    "heating_valve_stuck": {
        "description": "Heating valve is stuck in open or closed position",
        "severity": "medium",
        "actions": [
            "Verify valve actuator power supply",
            "Check valve actuator response to signal",
            "Inspect valve for debris or corrosion",
            "Try manual valve adjustment",
            "Check heating water flow rate",
            "Measure heating water temperature",
            "Replace valve actuator or valve if stuck"
        ]
    },
    "cooling_valve_stuck": {
        "description": "Cooling valve is stuck in open or closed position",
        "severity": "medium",
        "actions": [
            "Verify valve actuator power supply",
            "Check valve actuator response to signal",
            "Inspect valve for debris or corrosion",
            "Try manual valve adjustment",
            "Check cooling water flow rate",
            "Measure cooling water temperature",
            "Replace valve actuator or valve if stuck"
        ]
    },
    "damper_stuck": {
        "description": "Air damper is stuck in open or closed position",
        "severity": "medium",
        "actions": [
            "Verify damper actuator power supply",
            "Check damper actuator response to signal",
            "Inspect damper for debris or corrosion",
            "Try manual damper adjustment",
            "Check damper linkage for damage",
            "Measure air pressure across damper",
            "Replace damper actuator or damper if stuck"
        ]
    },
    "sensor_bias": {
        "description": "Sensor reading is consistently offset from actual value",
        "severity": "low",
        "actions": [
            "Verify sensor power supply",
            "Check sensor wiring connections",
            "Compare sensor reading to manual measurement",
            "Calibrate sensor if possible",
            "Check for sensor contamination",
            "Verify sensor is in correct location",
            "Replace sensor if defective"
        ]
    },
    "damper_oscillation": {
        "description": "Damper is oscillating rapidly instead of holding steady",
        "severity": "low",
        "actions": [
            "Check damper actuator for proper tuning",
            "Verify sensor signal is stable",
            "Inspect damper linkage for play or wear",
            "Check control loop proportional/integral gains",
            "Verify damper feedback signal",
            "Adjust control loop tuning parameters",
            "Replace damper actuator if defective"
        ]
    },
    "duct_imbalance": {
        "description": "Air distribution between ducts is imbalanced",
        "severity": "low",
        "actions": [
            "Verify damper positions in each duct",
            "Check for duct blockages or leaks",
            "Measure air flow in each duct",
            "Inspect duct damper actuators",
            "Check for duct damage or disconnection",
            "Balance damper positions",
            "Repair or replace ductwork if damaged"
        ]
    },
    "low_efficiency": {
        "description": "System is operating at lower than expected efficiency",
        "severity": "low",
        "actions": [
            "Check for air filter blockage",
            "Verify all fans are operating",
            "Check refrigerant charge level",
            "Inspect heat exchanger for fouling",
            "Verify thermostat calibration",
            "Check for duct leaks",
            "Schedule preventive maintenance"
        ]
    }
}


def get_playbook_actions(
    fault_type: str,
    top_k: int = None
) -> Dict[str, Any]:
    """
    Get recommended actions for a fault type
    
    Args:
        fault_type: Type of fault
        top_k: Limit to top K actions
        
    Returns:
        Dictionary with fault information and actions
    """
    if fault_type not in FAULT_PLAYBOOK:
        logger.warning(f"Unknown fault type: {fault_type}")
        return {
            "fault_type": fault_type,
            "description": "Unknown fault type",
            "severity": "unknown",
            "actions": ["Contact HVAC technician for inspection"]
        }
    
    fault_info = FAULT_PLAYBOOK[fault_type].copy()
    
    # Limit actions if requested
    if top_k and len(fault_info['actions']) > top_k:
        fault_info['actions'] = fault_info['actions'][:top_k]
    
    logger.info(f"Retrieved playbook for fault: {fault_type}")
    
    return fault_info


def get_all_fault_types() -> List[str]:
    """Get list of all known fault types"""
    return list(FAULT_PLAYBOOK.keys())


def save_playbook(
    output_path: str = "src/diagnostics/playbook.json"
) -> None:
    """
    Save playbook to JSON file
    
    Args:
        output_path: Path to save playbook
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(FAULT_PLAYBOOK, f, indent=2)
    
    logger.info(f"Saved playbook to {output_path}")


def load_playbook(
    input_path: str = "src/diagnostics/playbook.json"
) -> Dict[str, Any]:
    """
    Load playbook from JSON file
    
    Args:
        input_path: Path to playbook file
        
    Returns:
        Playbook dictionary
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.warning(f"Playbook file not found: {input_path}, using default")
        return FAULT_PLAYBOOK
    
    with open(input_path, 'r') as f:
        playbook = json.load(f)
    
    logger.info(f"Loaded playbook from {input_path}")
    
    return playbook
