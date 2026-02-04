# HVAC Fault Detection & Diagnostics Dataset Download Instructions

## Overview

The HVAC FDD Platform uses the **LBNL Fault Detection and Diagnostics Dataset**, which is publicly available and free to download. This dataset contains real building system telemetry from seven different HVAC equipment types, both in normal and faulted states.

## Dataset Information

- **Name**: LBNL Fault Detection and Diagnostics Data Sets
- **Source**: Lawrence Berkeley National Laboratory (LBNL)
- **Portal**: https://faultdetection.lbl.gov/data/
- **OpenEI Listing**: https://data.openei.org/submissions/5763
- **Scientific Paper**: https://www.nature.com/articles/s41597-023-02197-w
- **Format**: CSV files
- **Size**: ~500 MB (varies by equipment type)
- **License**: Public Domain / CC0

## Equipment Types Available

The dataset includes data from seven different HVAC systems:

1. **RTU (Rooftop Unit)** - Most common, recommended for starting
2. **Single-Duct AHU** - Air Handling Unit with single duct
3. **Dual-Duct AHU** - Air Handling Unit with dual ducts
4. **VAV** - Variable Air Volume terminal
5. **Fan Coil** - Fan coil unit
6. **Chiller Plant** - Chiller system
7. **Boiler Plant** - Boiler system

## Download Methods

### Method 1: Automated Download Script (Recommended)

```bash
# Navigate to project root
cd /path/to/hvac_fdd_platform

# Run the download script
python scripts/download_data.py --subsystem rtu

# Or download all subsystems
python scripts/download_data.py --all
```

### Method 2: Manual Download from LBNL Portal

1. Visit: https://faultdetection.lbl.gov/data/
2. Select your desired equipment type (e.g., RTU)
3. Download the CSV files
4. Extract to `data/raw/` directory

### Method 3: Download from OpenEI

1. Visit: https://data.openei.org/submissions/5763
2. Click "Download Full Dataset"
3. Extract the archive
4. Copy relevant CSV files to `data/raw/` directory

### Method 4: Using curl (Command Line)

```bash
# Example: Download RTU data
cd data/raw

# Download from LBNL FTP or HTTP endpoint
curl -O https://faultdetection.lbl.gov/data/rtu_data.csv

# Or use wget
wget https://faultdetection.lbl.gov/data/rtu_data.csv
```

## Directory Structure After Download

After downloading, your `data/raw/` directory should look like:

```
data/raw/
├── rtu_data.csv
├── rtu_metadata.csv
├── single_duct_ahu_data.csv
├── single_duct_ahu_metadata.csv
├── dual_duct_ahu_data.csv
├── dual_duct_ahu_metadata.csv
├── vav_data.csv
├── vav_metadata.csv
├── fan_coil_data.csv
├── fan_coil_metadata.csv
├── chiller_data.csv
├── chiller_metadata.csv
├── boiler_data.csv
└── boiler_metadata.csv
```

## Data Format

### CSV Structure

Each dataset CSV file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | UTC timestamp of the measurement |
| `point_name` | string | Name of the sensor/point (e.g., "supply_temp") |
| `value` | float | Measured value |
| `unit` | string | Unit of measurement (e.g., "°C", "Pa", "%") |
| `equipment_id` | string | ID of the equipment |
| `system_id` | string | ID of the system |
| `scenario_id` | string | ID of the fault scenario |
| `fault_id` | string | Fault type identifier |
| `fault_severity` | float | Severity level (if applicable) |

### Metadata CSV Structure

Metadata files contain:

| Column | Type | Description |
|--------|------|-------------|
| `point_name` | string | Sensor/point name |
| `description` | string | Human-readable description |
| `unit` | string | Unit of measurement |
| `data_type` | string | Data type (temperature, pressure, etc.) |
| `min_value` | float | Expected minimum value |
| `max_value` | float | Expected maximum value |
| `normal_range_min` | float | Normal operating minimum |
| `normal_range_max` | float | Normal operating maximum |

## Typical Fault Types

The dataset includes various fault scenarios:

### RTU Faults
- Supply fan failure
- Return fan failure
- Compressor failure
- Heating valve stuck
- Cooling valve stuck
- Sensor bias
- Duct imbalance

### AHU Faults
- Supply fan failure
- Return fan failure
- Heating valve stuck
- Cooling valve stuck
- Damper stuck
- Sensor bias

### VAV Faults
- Damper stuck
- Heating valve stuck
- Sensor bias
- Damper oscillation

### Chiller Faults
- Compressor failure
- Condenser fan failure
- Pump failure
- Sensor bias

### Boiler Faults
- Burner failure
- Pump failure
- Sensor bias
- Low efficiency

## Sample Data

For quick testing without downloading the full dataset, we provide a small sample dataset:

```bash
# Sample data is included in data/sample/
ls -la data/sample/

# Use sample data for testing
python -m src.modeling.train --config configs/train.yaml --data-path data/sample/
```

## Data Validation

After downloading, validate your data:

```bash
# Check data integrity
python scripts/validate_data.py --path data/raw/

# Generate data summary report
python scripts/data_summary.py --path data/raw/
```

## Storage Requirements

| Subsystem | Compressed | Uncompressed |
|-----------|-----------|--------------|
| RTU | ~50 MB | ~150 MB |
| Single-Duct AHU | ~40 MB | ~120 MB |
| Dual-Duct AHU | ~45 MB | ~135 MB |
| VAV | ~35 MB | ~100 MB |
| Fan Coil | ~30 MB | ~90 MB |
| Chiller | ~60 MB | ~180 MB |
| Boiler | ~50 MB | ~150 MB |
| **All** | ~310 MB | ~925 MB |

**Recommendation**: Start with RTU (~150 MB uncompressed) for initial development.

## Troubleshooting

### Download Fails

```bash
# Check internet connection
ping faultdetection.lbl.gov

# Try alternative mirror
wget --mirror https://data.openei.org/submissions/5763

# Use curl with verbose output
curl -v https://faultdetection.lbl.gov/data/rtu_data.csv
```

### Corrupted Files

```bash
# Verify file integrity
md5sum data/raw/*.csv

# Check file size
du -h data/raw/

# Validate CSV format
python -c "import pandas as pd; pd.read_csv('data/raw/rtu_data.csv', nrows=5)"
```

### Missing Columns

```bash
# Check available columns
python -c "import pandas as pd; df = pd.read_csv('data/raw/rtu_data.csv', nrows=1); print(df.columns.tolist())"

# Compare with expected schema
python scripts/validate_data.py --path data/raw/ --verbose
```

## Data Privacy & Citation

When using this dataset, please:

1. **Cite the paper**: Granderson, J., et al. (2023). "LBNL Fault Detection and Diagnostics Dataset." Scientific Data, 10, 197.

2. **Acknowledge the source**: Lawrence Berkeley National Laboratory

3. **Follow the license**: The dataset is in the public domain (CC0)

## Additional Resources

- **LBNL FDD Portal**: https://faultdetection.lbl.gov/
- **Scientific Paper**: https://www.nature.com/articles/s41597-023-02197-w
- **Brick Schema**: https://brickschema.org/
- **OpenEI**: https://openei.org/

## Next Steps

After downloading the dataset:

1. Validate the data: `python scripts/validate_data.py`
2. Generate summary statistics: `python scripts/data_summary.py`
3. Train the model: `make train`
4. Evaluate the model: `make evaluate`
5. Run inference: `make api`

## Support

For issues with dataset access:
- Check LBNL website: https://faultdetection.lbl.gov/
- Review OpenEI documentation: https://data.openei.org/
- Check dataset paper: https://www.nature.com/articles/s41597-023-02197-w

---

**Last Updated**: February 2026
**Dataset Version**: 1.0
**Platform Version**: 1.0.0
