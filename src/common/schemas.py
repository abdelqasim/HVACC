"""
Pydantic schemas for HVAC FDD Platform
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class FaultTicket(BaseModel):
    """Fault ticket schema"""
    
    ticket_id: str = Field(..., description="Unique ticket identifier")
    timestamp: datetime = Field(..., description="Timestamp of fault detection")
    fault_present: bool = Field(..., description="Whether fault is present")
    fault_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of fault detection")
    fault_type: Optional[str] = Field(None, description="Type of fault detected")
    fault_type_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence of fault classification")
    severity_score: float = Field(..., ge=0.0, le=1.0, description="Severity score (0-1)")
    severity_level: str = Field(..., description="Severity level (low, medium, high)")
    
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Evidence supporting diagnosis")
    recommended_checks: List[str] = Field(default_factory=list, description="Recommended technician checks")
    
    model_version: str = Field(..., description="Model version used for prediction")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Ticket creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "ticket_20230115_001",
                "timestamp": "2023-01-15T10:30:00Z",
                "fault_present": True,
                "fault_confidence": 0.92,
                "fault_type": "supply_fan_failure",
                "fault_type_confidence": 0.87,
                "severity_score": 0.78,
                "severity_level": "high",
                "evidence": {
                    "top_features": ["supply_fan_speed", "supply_pressure"],
                    "feature_contributions": [0.35, 0.28],
                },
                "recommended_checks": ["Verify supply fan motor operation"],
                "model_version": "v1.0.0",
                "created_at": "2023-01-15T10:35:00Z"
            }
        }


class PredictionRequest(BaseModel):
    """Single prediction request schema"""
    
    timestamp: datetime = Field(..., description="Timestamp of the window")
    data: Dict[str, List[float]] = Field(..., description="Windowed telemetry data")
    subsystem: str = Field(default="rtu", description="HVAC subsystem type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2023-01-15T10:30:00Z",
                "data": {
                    "supply_temp": [68.5, 68.3, 68.1],
                    "return_temp": [72.1, 72.0, 71.9],
                    "supply_fan_speed": [0.85, 0.85, 0.84]
                },
                "subsystem": "rtu"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    
    csv_path: Optional[str] = Field(None, description="Path to CSV file")
    subsystem: str = Field(default="rtu", description="HVAC subsystem type")
    window_size_minutes: int = Field(default=30, description="Window size in minutes")
    stride_minutes: int = Field(default=5, description="Stride in minutes")
    
    class Config:
        json_schema_extra = {
            "example": {
                "csv_path": "data/sample.csv",
                "subsystem": "rtu",
                "window_size_minutes": 30,
                "stride_minutes": 5
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""
    
    tickets: List[FaultTicket] = Field(..., description="List of fault tickets")
    total_windows: int = Field(..., description="Total number of windows processed")
    faults_detected: int = Field(..., description="Number of faults detected")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tickets": [],
                "total_windows": 100,
                "faults_detected": 5,
                "processing_time_seconds": 2.5
            }
        }


class ExplanationRequest(BaseModel):
    """Explanation request schema"""
    
    ticket_id: str = Field(..., description="Ticket ID to explain")
    include_plots: bool = Field(default=True, description="Include evidence plots")
    include_shap: bool = Field(default=True, description="Include SHAP values")
    top_k_features: int = Field(default=10, description="Number of top features to show")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "ticket_20230115_001",
                "include_plots": True,
                "include_shap": True,
                "top_k_features": 10
            }
        }


class ExplanationResponse(BaseModel):
    """Explanation response schema"""
    
    ticket_id: str = Field(..., description="Ticket ID")
    top_features: List[str] = Field(..., description="Top contributing features")
    feature_contributions: List[float] = Field(..., description="Feature contribution values")
    shap_values: Optional[Dict[str, float]] = Field(None, description="SHAP values")
    anomalous_signals: List[str] = Field(..., description="Signals showing anomalies")
    evidence_plots: List[str] = Field(default_factory=list, description="Paths to evidence plots")
    playbook_actions: List[str] = Field(default_factory=list, description="Recommended actions from playbook")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticket_id": "ticket_20230115_001",
                "top_features": ["supply_fan_speed", "supply_pressure"],
                "feature_contributions": [0.35, 0.28],
                "anomalous_signals": ["supply_fan_speed"],
                "evidence_plots": ["plot_1.png", "plot_2.png"],
                "playbook_actions": ["Verify supply fan motor operation"]
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    mlflow_connected: bool = Field(..., description="Whether MLflow is connected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "mlflow_connected": True
            }
        }


class DataValidationError(BaseModel):
    """Data validation error schema"""
    
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid input data",
                "details": {"missing_columns": ["supply_temp"]}
            }
        }
