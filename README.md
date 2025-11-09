# Low-Power-Neuromorphic-AI-for-Satellite-Anomaly-Detection
Low-Power Neuromorphic AI for Satellite Anomaly Detection

This project compares Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs) for satellite telemetry anomaly detection, focusing on accuracy vs. energy efficiency.

The goal is to demonstrate that neuromorphic (SNN-based) models can achieve similar performance to conventional ANNs while consuming significantly less power, making them ideal for on-board satellite AI systems.

# Overview

Satellites continuously generate vast telemetry data that can indicate system health or potential failures.
Running AI models onboard is challenging due to limited energy availability.
This project explores Spiking Neural Networks (SNNs) — brain-inspired architectures that process information through discrete spikes — to drastically reduce energy consumption while maintaining detection accuracy.

# Methodology

Dataset: Satellite telemetry dataset containing normal and anomalous readings.

# Models Trained:

A baseline ANN using traditional dense layers.

A SNN version trained and simulated using spiking neuron models.

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, and Relative Energy Consumption.

# Tools Used:

Python

PyTorch

NumPy, Pandas, Matplotlib

# Results
Model	Accuracy	Precision	Recall	F1 Score	Energy (Relative Units)
ANN	95.76%	95.71%	95.71%	95.71%	163,200
SNN	94.82%	94.74%	94.75%	94.75%	9.8

Result: SNN achieves nearly identical accuracy to ANN while reducing energy usage by over 16,000×.

# Conclusion

SNNs can maintain strong anomaly detection performance while being drastically more energy-efficient.

This demonstrates the potential of neuromorphic AI for edge and satellite applications, where power and resources are limited.

# Future Work

Explore on-chip deployment using neuromorphic hardware (e.g., Intel Loihi, SpiNNaker).

Extend anomaly detection to multi-satellite or multi-sensor systems.

Experiment with spike encoding strategies and hybrid ANN–SNN models.
