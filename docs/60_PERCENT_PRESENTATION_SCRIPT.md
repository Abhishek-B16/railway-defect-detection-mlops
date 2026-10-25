# RailGuard Presentation Script (Word-for-Word Speaking Notes)

---

## 1. Introduction & Problem Statement (1 Minute)

"Good morning, Professor.

Today, I am presenting the 60% completion milestone of our project: RailGuard.

In India and across the world, railway track inspection is still mostly done by manual walking patrol. This is slow, risky, and humans can easily miss hairline cracks due to fatigue.

Our goal is not just to train a basic AI model in a Jupyter Notebook. A model alone is only 10% of a real solution. What happens when the model makes a mistake, or when lighting changes, or when we need to deploy it on real inspection trains?

To solve this, we built a complete, production-grade MLOps system. It automatically versions our track data, tracks all training experiments, deploys certified models safely through a Model Registry, and monitors live performance with real-time dashboards."

---

## 2. Our Architecture (How It Works) (1 Minute)

"Our project connects four core systems together:

First, DVC and DagsHub: We version-control our railway crack images and annotations just like Git tracks code.

Second, MLflow: We log every training run, every hyperparameter, and every accuracy metric to the cloud.

Third, MLflow Model Registry: Our production server never touches raw training files. It only loads whichever model has the certified 'candidate' tag.

Fourth, Docker and FastAPI: The application runs inside Docker containers with a FastAPI backend and a React user interface. At the same time, Prometheus and Grafana collect real-time speed and health metrics."

---

## 3. How We Evolved Our Models (V1 to V5) (1.5 Minutes)

"We did not stop at one model. We tested and improved five iterations:

We started with V1 and V2 using YOLOv8 Nano. It was very fast, but accuracy was only around 38% mAP50 because Nano is too small to catch fine, thin cracks.

In V3, we cleaned our dataset, removed duplicate overlapping boxes, and boosted accuracy to 47%.

In V4, we stepped up to YOLOv8 Medium with 25 million parameters. It reached 50% mAP50 and converged very smoothly over 40 epochs.

Finally, in V5, we moved to YOLOv8 Large with 1024-pixel high-definition image tiling. Because the resolution is higher and the model has more capacity, V5 achieved our highest peak accuracy of 52.2% mAP50, and over 31% strict mAP50-95.

This V5 model is the one currently running in our live system."

---

## 4. Live Demonstration (Show Your Screen) (3 Minutes)

### Screen 1: The Live Web App (localhost:5173)

(Action: Open React UI and upload a crack image)

"Here is our live diagnostic portal. When I upload this railway image, our containerized FastAPI backend runs inference in less than 40 milliseconds. It draws a clean bounding box right over the crack, calculates a confidence score, and tells us the safety severity.

(Action: Upload a clean track image)

When I upload an image of a clean track with no defects, it correctly identifies that the track is clear and outputs: 'No Defects Found! Track Status: Safe'."

---

### Screen 2: Real-Time Grafana Monitoring (localhost:3001)

(Action: Switch tab to Grafana)

"Now, let us look behind the scenes at production monitoring.

Every time someone uploads an image, the backend sends telemetry into Prometheus and displays it here in Grafana.

We track three key things:
1. Prediction request rate to see live API traffic.
2. Inference latency, which stays below 50 milliseconds, proving this can run in real time on moving trains.
3. Defect class counts, so engineers immediately get alerted if defect rates spike unexpectedly."

---

### Screen 3: MLflow Experiment Tracking (DagsHub MLflow)

(Action: Switch tab to MLflow Experiments and show the comparison curves)

"Here inside MLflow, we have proof of our model evolution.

Instead of keeping numbers in spreadsheets, MLflow plots our mAP50 and loss curves epoch by epoch.

You can clearly see our Nano baseline in purple plateauing early, our Medium model in green providing steady learning, and our Large V5 model in yellow achieving the highest peak accuracy."

---

### Screen 4: Model Registry

(Action: Click Models -> RailGuard-Detector)

"Finally, this is our Model Registry.

You can see all versions from 1 to 5. Version 5 holds the 'candidate' alias. Our FastAPI service dynamically pulls whatever model holds this alias. This means we can deploy a new model in production without stopping or restarting the server."

---

## 5. What We Completed & What Is Left (1 Minute)

"To summarize:

What is done (60% milestone):
- End-to-end dataset versioning with DVC.
- Iterative model training from V1 to V5 on YOLOv8.
- Experiment tracking and Model Registry on MLflow.
- Containerized microservices using Docker Compose and FastAPI.
- Modern React diagnostic UI.
- Real-time observability dashboard with Prometheus and Grafana.

What is planned for the final 40%:
- GitHub Actions CI/CD to automatically retrain the model whenever new data is added.
- Automated data drift detection with Evidently AI.
- Model quantization using ONNX or TensorRT for low-power edge devices on train engines.

Thank you, Professor. I am ready for any questions."

---

## Quick Answers for Common Questions:

Q: Why use MLflow instead of training on your PC?
A: MLflow gives complete reproducibility. It saves the exact code commit, parameters, weights, and metrics in the cloud so anyone can verify and deploy the exact same model.

Q: Why is Grafana on port 3001?
A: In our Docker Compose file, we mapped port 3001 to prevent conflicts with other local services using port 3000.

Q: What is mAP50?
A: Mean Average Precision at 50% overlap. It measures whether the AI found the crack in the right place with at least 50% box overlap compared to human labels.
