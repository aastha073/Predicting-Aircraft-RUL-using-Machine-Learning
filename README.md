# Predicting-Aircraft-RUL-using-Machine-Learning

# Aircraft Turbofan Engine RUL Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS-orange)](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

## 📋 Overview

This project implements and compares multiple machine learning algorithms to predict the Remaining Useful Life (RUL) of aircraft turbofan engines using NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The implementation includes custom-built algorithms from scratch alongside scikit-learn implementations, with a special focus on the clipping approach that achieved the best performance.

## 🎯 Project Objectives

- Implement and compare multiple ML algorithms (both from scratch and using scikit-learn)
- Apply comprehensive feature engineering and data preprocessing techniques
- Develop a clipping threshold approach for improved RUL predictions
- Achieve high accuracy in predicting engine failure before it occurs
- Provide detailed visualizations of sensor behavior over engine lifetime

## 🚀 Key Features

- **Custom ML Implementations**: Linear Regression, Decision Trees, Random Forest, and SVR built from scratch
- **Comprehensive Data Preprocessing**: Removal of constant features, handling highly correlated features
- **Advanced Feature Engineering**: RUL calculation, sensor data normalization, correlation analysis
- **Clipping Threshold Strategy**: Novel approach setting RUL threshold at 125 cycles
- **Extensive Visualizations**: Sensor behavior plots, correlation heatmaps, failure time distributions
- **Best Model Performance**: SVM with clipping threshold achieving 77.82% R² Score

## 📊 Dataset

The project uses NASA's Prognostics Data Repository FD001 turbofan engine degradation simulation dataset:

- **Training Data**: `train_FD001.txt` - 20,631 samples from 100 engines
- **Test Data**: `test_FD001.txt` - 13,096 samples
- **RUL Ground Truth**: `RUL_FD001.txt` - Actual RUL values for test engines

### Sensor Measurements
The dataset includes 21 sensor readings:
- **Temperature Sensors**: Fan inlet, LPC outlet, HPC outlet, LPT outlet temperatures
- **Pressure Sensors**: Fan inlet, bypass-duct, HPC outlet pressures
- **Speed Sensors**: Physical fan speed, corrected fan/core speeds
- **Other Sensors**: Bypass ratio, fuel flow ratio, cooling air flow, etc.

### Data Characteristics
- Time-series data with operational cycles
- 3 operational settings per cycle
- Engines run from healthy state until failure
- RUL calculated as cycles remaining until failure


## 🔧 Implementation Details

### Data Preprocessing Pipeline

1. **Feature Removal**:
   - Removed constant features (std < 0.02): `op_setting_3`, `sm_1`, `sm_5`, `sm_6`, `sm_10`, `sm_16`, `sm_18`, `sm_19`
   - Removed highly correlated feature: `sm_9` (96% correlation with `sm_14`)

2. **RUL Calculation**:
   ```python
   df['rul'] = df.groupby('engine')['time'].transform('max') - df['time']
   ```

3. **Data Normalization**:
   - MinMaxScaler applied to scale features between 0 and 1

### Custom ML Implementations

The project includes from-scratch implementations of:
- **Linear Regression**: Gradient descent optimization
- **Decision Tree**: Gini impurity splitting criterion
- **Random Forest**: Bootstrap aggregation with 10 trees
- **Support Vector Regression**: Epsilon-insensitive loss

### Clipping Strategy

A critical insight was implementing RUL clipping at 125 cycles:
- Mean failure time: 205 cycles (std: 46)
- Clipping threshold: 125 cycles
- Rationale: Engines with RUL > 125 are considered healthy, focusing model attention on critical maintenance periods

## 📈 Results

### Model Performance Comparison

| Model | Implementation | Train RMSE | Test RMSE | Validation RMSE | Validation R² |
|-------|----------------|------------|-----------|-----------------|---------------|
| Linear Regression | From Scratch | 47.37 | 47.02 | 31.91 | 41.05% |
| Decision Tree | From Scratch | 48.52 | 47.96 | 31.61 | 42.13% |
| Random Forest | From Scratch | 47.53 | 46.53 | 22.97 | 69.45% |
| SVR | From Scratch | 65.56 | 64.22 | 44.00 | -12.09% |
| SVM | Sklearn | 42.91 | 42.25 | 26.15 | 60.41% |
| **SVM (Clipped)** | **Sklearn** | **-** | **-** | **19.57** | **77.82%** |

### Key Findings
- SVM with RUL clipping at 125 cycles achieved the best performance
- Feature engineering significantly improved all models
- Removing highly correlated features improved model stability
- Custom implementations provided valuable insights into algorithm mechanics

### Visualizations
The notebook includes:
- Failure time distribution across 100 engines
- Correlation heatmap showing sensor relationships
- Time-series plots of sensor readings vs RUL
- Model prediction scatter plots

## 🔍 Methodology

### Feature Engineering Process
1. **Constant Feature Detection**: Removed features with standard deviation < 0.02
2. **Correlation Analysis**: Identified and removed features with >95% correlation
3. **RUL Derivation**: Calculated from maximum operational time per engine
4. **Feature Scaling**: MinMaxScaler normalization for all sensor readings

### Model Development Approach
1. **Baseline Models**: Implemented algorithms from scratch to understand mechanics
2. **Sklearn Comparison**: Validated custom implementations against library versions
3. **Hyperparameter Tuning**: GridSearchCV for SVM optimization (C values: 5-50)
4. **Clipping Innovation**: Applied domain knowledge to improve predictions

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Primary metric for prediction accuracy
- **R² Score**: Percentage of variance explained by the model
- **Train/Test/Validation Split**: 80/20 split with separate validation set

## 📚 Code Structure

The Jupyter notebook is organized into the following sections:

1. **Data Loading and Exploration**
   - Loading FD001 dataset files
   - Initial data exploration and sensor dictionary creation

2. **Data Preprocessing**
   - Constant feature removal
   - Correlation analysis and feature selection
   - RUL calculation

3. **Exploratory Data Analysis**
   - Failure time distribution
   - Sensor behavior visualization over RUL
   - Correlation heatmap

4. **Model Implementation**
   - Custom Linear Regression with gradient descent
   - Custom Decision Tree with Gini impurity
   - Custom Random Forest with bootstrap aggregation
   - Custom SVR with epsilon-insensitive loss
   - Sklearn SVM comparison

5. **Model Evaluation**
   - Performance metrics calculation
   - Hyperparameter tuning with GridSearchCV
   - Clipping threshold optimization

## 📊 Key Visualizations

- **Failure Time Bar Chart**: Shows RUL distribution across 100 engines
- **Correlation Heatmap**: Reveals relationships between 21 sensors
- **Sensor Time Series**: Plots sensor readings against RUL for pattern analysis
- **Prediction Scatter Plot**: Compares predicted vs actual RUL values

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Implementation of deep learning models (LSTM, GRU)
- Additional feature engineering techniques
- Deployment pipeline development
- Performance optimization


## 📖 References

1. Melkumian, Stanley A. (2024). "Predictive Maintenance Analysis of Turbofan Engine Sensor Data," The Journal of Purdue Undergraduate Research
2. NASA Prognostics Data Repository - Turbofan Engine Degradation Simulation Dataset
3. Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2020). "Predictive Maintenance of Turbofan Engines"
4. Peters (2023). Exploratory Data Analysis for RUL Prediction

## 🔮 Future Work

- **Deep Learning Implementation**: LSTM/GRU networks for sequential pattern learning
- **Real-time Dashboard**: Streamlit application for live RUL monitoring
- **Multi-fault Detection**: Extend to identify specific failure modes
- **Transfer Learning**: Apply models to other engine types (FD002, FD003, FD004)
- **Ensemble Methods**: Combine multiple models for improved robustness
- **Feature Importance Analysis**: SHAP values for model interpretability

---

**Note**: This project demonstrates the practical application of machine learning for predictive maintenance in aviation, showcasing both theoretical understanding through custom implementations and practical results through optimized models.
