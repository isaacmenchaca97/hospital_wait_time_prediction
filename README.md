# Hospital Wait Time Prediction - Data Cleaning with SageMaker

This project implements a data cleaning pipeline for hospital wait time prediction using Amazon SageMaker. The implementation leverages various SageMaker features including Processing Jobs, Feature Store, Data Quality Monitoring, and Random Cut Forest for outlier detection.

## Prerequisites

- AWS Account with appropriate permissions
- Python 3.7+
- AWS CLI configured
- Required Python packages (install using `pip install -r requirements.txt`):
  - boto3
  - sagemaker
  - pandas
  - numpy
  - scikit-learn

## Project Structure

```
.
├── README.md
├── requirements.txt
├── sagemaker_data_cleaning.py    # Main SageMaker pipeline implementation
└── preprocessing_script.py       # Data preprocessing script for SageMaker Processing Jobs
```

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Configure your AWS credentials:
```bash
aws configure
```

3. Create an S3 bucket for storing data and artifacts:
```bash
aws s3 mb s3://your-bucket-name
```

4. Create an IAM role with the following permissions:
   - AmazonSageMakerFullAccess
   - AmazonS3FullAccess
   - AWSCloudFormationFullAccess

## Usage

1. Update the AWS configuration in `sagemaker_data_cleaning.py`:
```python
role_arn = 'your-role-arn'
bucket_name = 'your-bucket-name'
```

2. Upload your raw data to S3:
```bash
aws s3 cp data.csv s3://your-bucket-name/raw-data/
```

3. Run the data cleaning pipeline:
```bash
python sagemaker_data_cleaning.py
```

## Pipeline Components

### 1. Data Processing
- Removes duplicate records
- Handles missing values
- Processes date features
- Detects outliers using z-score method
- Encodes categorical variables
- Scales numerical features

### 2. Feature Store
- Creates a Feature Group for storing processed features
- Enables online and offline feature access
- Maintains feature versioning

### 3. Data Quality Monitoring
- Monitors data quality metrics
- Generates alerts for data drift
- Tracks feature statistics

### 4. Outlier Detection
- Uses SageMaker Random Cut Forest algorithm
- Identifies anomalies in numerical features
- Flags potential data quality issues

## Output

The pipeline produces the following outputs in your S3 bucket:

1. Processed Data:
   - `s3://your-bucket-name/processed-data/processed_data.csv`
   - `s3://your-bucket-name/processed-data/processing_summary.json`

2. Feature Store:
   - `s3://your-bucket-name/feature-store/`

3. Monitoring Results:
   - `s3://your-bucket-name/monitoring/`

## Monitoring and Maintenance

1. Monitor data quality:
   - Access CloudWatch metrics for data quality
   - Review monitoring reports in SageMaker Studio

2. Update feature definitions:
   - Modify `preprocessing_script.py` for new features
   - Update Feature Store schema as needed

3. Pipeline maintenance:
   - Adjust processing resources based on data volume
   - Update monitoring thresholds as needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

