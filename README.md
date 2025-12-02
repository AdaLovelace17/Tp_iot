# IoT Smart Grid Energy Management System

## Executive Summary
This project implements an IoT-based Smart Grid Energy Management System with real-time monitoring, cloud storage, and AI-powered energy consumption prediction. The system integrates hardware sensors (virtualized as Python scripts), MQTT communication protocol, MongoDB Atlas for cloud storage, a Django-based dashboard for visualization, and a Random Forest ML model for energy forecasting.

---

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Project Requirements](#project-requirements)
3. [Installation](#installation)
4. [Database Connection](#database-connection)
5. [Running the Dashboard](#running-the-dashboard)
6. [Running Sensors and Gateways](#running-sensors-and-gateways)
7. [Project Structure](#project-structure)

---

## System Architecture Overview
The system is organized in multiple layers:

- **Hardware Layer**: IoT sensors and smart meters (Python scripts) publishing data via MQTT.
- **Communication Layer**: MQTT broker (localhost) for publish/subscribe messaging.
- **Gateway Layer**: Regional gateways for data aggregation, enrichment, and local storage.
- **Cloud Layer**: `Cloud_receiver.py` subscribes to all topics and writes data to MongoDB Atlas.
- **Backend Layer**: Django application serving a dynamic dashboard and API endpoints for AI predictions.
- **AI/ML Module**: `tain_ai.py` performs feature engineering, trains a Random Forest model, and saves artifacts for predictions.

**Data Flow:**  
Sensor → MQTT Broker → Regional Gateway → Cloud Receiver → MongoDB Atlas → Dashboard & AI Module

---


---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/AdaLovelace17/Tp_iot.git
cd Tp_iot
