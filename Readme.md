# UK Student Attainment Predictor — MLOps Pipeline

This project delivers a full end‑to‑end machine‑learning pipeline that trains a Logistic Regression model to predict a 6th‑form student’s A‑Level Maths grade based on their prior attainment, including SATS scores, GCSE grades, and GCE AS‑level results. The trained model is then deployed for live inference using NVIDIA Triton Inference Server running on a local Kubernetes cluster.

To support the model the dataset of 10,000 rows of purely synthetic student data. (The dataset included in this repo was generated using synthetic_uk_attainment_10000.ipynb. It is pre-built and ready to use — no generation step is required.) The dataset reflects UK national averages for 2024/25, using normally distributed values for SATS (80–120), GCSE grades (1–9), and GCE AS grades (A–U). Each record also includes an A-Level grade that is generated independently during synthetic data creation and then used as the training label. All generated data is saved as a raw CSV file.

System Requirements table — Docker Desktop, Minikube, kubectl, k6, and an NGC account. (Developed and tested on Ubuntu Desktop 24.04 Linux.).

9 numbered steps in strict order:

1 Start Minikube with enough GPU/RAM
2 Apply **`first_deployment.yaml`** — spins up MinIO + JupyterLab
3 Apply **`second_deployment.yaml`** — creates the datasets and Triton-model-repo buckets
— Check for all processes to be running
4 Upload the CSV File to the MinIO Bucket via the web console (WebUI) (Check if there are in another Bucket files: config.pbtxt and model.onnx- delete them, leftovers from previous CPU run)
—  Open JupyterLab in a browser

5 Upload and run the notebook (not locally on the device)— with a table explaining what each section does and what a successful upload looks like

6 Open the **`triton_mathmodel_k6_test.js`** in VSCode and align **`TRITON_HOST`** and **`TRITON_PORT`** it must be the same as shows in the terminal.

7 Apply **`third_deployment_gpu.yaml`** — deploys Triton, which auto-loads the model from MinIO

8 Verify Triton health and fire a test inference with curl

9 Run the k6 load test with the correct environment variables

It also includes a grade reference table, expected k6 thresholds, and a Troubleshooting section covering the most common failure points (PVC errors, image pull failures, MinIO path mismatches, and the important note that the notebook must run inside the cluster — not locally).

---

## How This Project Works (Big Picture)

```
CSV Dataset (MinIO)
       ↓
JupyterLab Notebook
  • Exploratory Data Analysis (EDA) & encoding
  • Train Logistic Regression
  • Export → model.onnx + config.pbtxt
  • Upload artefacts back to MinIO
       ↓
Triton Inference Server
  • Loads model.onnx from MinIO on startup
  • Serves predictions over HTTP on port 8000
       ↓
k6 Load Test
  • Fires 1,000 inference requests
  • Measures latency + error rates
```

---

## System Requirements

Before you start, install all of the following on your machine.
| Tool | Version | Purpose |
|------|---------|---------|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Latest | Container runtime |
| [Minikube](https://minikube.sigs.k8s.io/docs/start/) | v1.32+ | Local Kubernetes cluster |
| [kubectl](https://kubernetes.io/docs/tasks/tools/) | v1.29+ | Kubernetes CLI |
| [k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) | Latest | Load testing tool |
| [NVIDIA Driver](https://www.nvidia.com/en-us/drivers/) | ≥ 525 | Required for CUDA 12.x compatibility |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) | ≥ 12.x | Required by `tritonserver:26.02-py3` |
| [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Latest | Allows Docker to pass the GPU through to containers |
| [NGC Account](https://ngc.nvidia.com/signin) | — | Required to pull images from `nvcr.io` |

> **Note for Linux users:** This project was developed on Ubuntu 24.04. Use **Docker Engine** installed via `apt`, not Docker Desktop (which is a macOS/Windows product).

> **Hardware:** A CUDA-capable NVIDIA GPU with **≥ 8 GB VRAM** is required. The Triton deployment uses the image (`tritonserver:26.02-py3`).requires CUDA 12.x and NVIDIA driver ≥ 525.
> **RAM:** Allocate at least **10 GB** to Minikube (MinIO needs 2 GB, JupyterLab 2 GB, Triton 4 GB).

---

## Project Structure

```
GPU_Project_Repo/
├── Data/
│   └── synthetic_uk_attainment_10000_clean_1.csv   # 10,000-row synthetic dataset
├── Notebooks/
│   └── FIXED_Completed_Project.ipynb               # Training & export notebook
├── K8s_Manifests/
│   ├── first_deployment.yaml    # Stage 1 — Namespace, MinIO, JupyterLab
│   ├── second_deployment.yaml   # Stage 2 — Bucket init job
│   ├── upload the dataset to triton    # Stage 2 — Triton Bucket
│   ├── check if there are files: config.pbtxt and model.onnx- delete them    # Stage 2 — Triton Bucket- lleftovers from previous CPU run
│   ├── upload and run - notebook to Jupyter lab FIXED_Completed_Project.ipynb    # Stage 2 — load Jupyter lab through the Minio environment
│   ├── third_deployment_gpu.yaml    # Stage 3 — Triton Inference Server
└── K6_Test_Scripts/
    └── triton_mathmodel_k6_test.js  # Load test (1,000 inference requests)
```

---

## Step-by-Step Instructions

### Step 1 — Start Minikube

Open a terminal and start a local Kubernetes cluster with enough resources. The `--gpus all` flag is required to make the GPU visible inside the cluster:

```bash
minikube start \
  --cpus=4 \
  --memory=10240 \
  --disk-size=30g \
  --driver=docker \
  --container-runtime=docker \
  --gpus all
```

Verify the cluster is running:

```bash
kubectl get pods -A -w
# Expected: one node with STATUS = Ready
```
Verify the GPU is visible to Kubernetes:

```bash
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
# Expected: minikube   1
---

### Step 2 — Deploy Stage 1: Namespace, MinIO & JupyterLab

This manifest creates:
- The `mlops` namespace
- MinIO object storage (data + model artefact store)
- JupyterLab (where the notebook runs)
- Required ConfigMaps, Secrets, PVCs, and RBAC roles

```bash
kubectl apply -f K8s_Manifests/first_deployment.yaml
```

Wait for all pods to be running:

```bash
kubectl get pods -n mlops -w
# Wait until both `minio-...` and `jupyter-lab-...` show STATUS = Running 1/1
```

This may take a few minutes the first time as Container images are pulled (downloaded).

---

### Step 3 — Deploy Stage 2: Initialise MinIO Buckets

This runs a one-off Kubernetes Job that connects to MinIO and creates two buckets:
- `datasets` — where the CSV will be uploaded
- `triton-model-repo` — where the trained model artefacts will be stored

```bash
kubectl apply -f K8s_Manifests/second_deployment.yaml
```

Confirm the job completed successfully:

```bash
kubectl get jobs -n mlops -w
# Expected: minio-init-bucket   1/1   Completed
```

If it stays in `0/1`, check the logs:

```bash
kubectl logs -n mlops job/minio-init-bucket
```

---

### Step 4 — Upload the Dataset to MinIO

Open the MinIO Console in your browser:

```bash
minikube service minio-webui -n mlops --url
# e.g. http://192.168.49.2:YYYYY
```

Log in with:
- **Username:** `minioadmin`
- **Password:** `minioadminpassword`

Then:
1. Navigate to the **`datasets`** bucket.

2. Click **Upload** and select `Data/synthetic_uk_attainment_10000_clean_1.csv`.

3. Make sure the file appears at the root of the bucket (not inside a subfolder) so the notebook can find it at `datasets/synthetic_uk_attainment_10000_clean_1.csv`.
> **If this is not your first run:** navigate to the **`triton-model-repo`** bucket and check for any leftover `config.pbtxt` and `model.onnx` files from a previous CPU run. Delete them before proceeding.
---

### Step 5 — Open JupyterLab

Get the JupyterLab URL:

```bash
minikube service jupyter-lab -n mlops --url
# e.g. http://192.168.49.2:30001
```
Open that URL in your browser. No password is required (token authentication is disabled).

---

### Step 6 — Upload and Run the Notebook

1. In JupyterLab, click the **Upload** button (↑ arrow in the file browser) and upload `Notebooks/FIXED_Completed_Project.ipynb`.

2. Open the notebook.

3. Run all cells **in order** from top to bottom using **Run → Run All Cells** (or `Shift+Enter` cell by cell).

**What each section of the notebook does:**

| Notebook Section | What It Does |
|-----------------|-------------|
| **Install packages** | Installs pandas, scikit-learn, skl2onnx, onnxruntime, s3fs, etc. |
| **Connect to MinIO** | Connects to MinIO at `minio-api.mlops.svc.cluster.local:9000` and loads the CSV |
| **Grade mappings** | Defines the 0–6 ordinal scale (U=0, E=1, D=2, C=3, B=4, A=5, A*=6) |
| **EDA** | Prints distributions and renders a pairplot of all attainment variables |
| **Encode & split** | Encodes letter grades to numbers, splits 80/20 train/test with stratification |
| **Train model** | Fits a `StandardScaler → LogisticRegression(lbfgs)` sklearn Pipeline |
| **Evaluate** | Prints accuracy, classification report, and confusion matrices (7-class + pass/fail) |
| **Save .pkl** | Saves the sklearn pipeline to `model.pkl` |
| **Convert to ONNX** | Serializes `model.pkl` → `model.onnx`, renames output tensor to `output_label` |
| **Validate ONNX** | Runs a test inference locally with `onnxruntime` to confirm the model works |
| **Generate config.pbtxt** | Writes the Triton model configuration file |
| **Upload to MinIO** | Uploads `config.pbtxt` and `model.onnx` to `triton-model-repo/models/mathmodel/` |

After the notebook completes you should see a confirmation like:

```
✓ All artefacts uploaded to MinIO successfully.
  triton-model-repo/models/mathmodel/config.pbtxt  (...)
  triton-model-repo/models/mathmodel/1/model.onnx  (...)
  
 Optionally verify the artefacts are present in MinIO:

```bash
minikube service minio-webui -n mlops --url
# Log in and navigate to triton-model-repo → models → mathmodel
# You should see: config.pbtxt and 1/model.onnx

---
### Step 7 — Configure the k6 Test Script

Before deploying Triton, note the Minikube IP and the Triton NodePort so you can set them in the load test script.

Get the Minikube IP:

```bash
minikube ip
# e.g. 192.168.49.2
```

The Triton NodePort will be visible after Step 8 (`kubectl get svc -n mlops triton-http`). Open `K6_Test_Scripts/triton_mathmodel_k6_test.js` in your editor and ensure the `TRITON_HOST` and `TRITON_PORT` values will match — or pass them as environment variables at runtime as shown in Step 9.

---
### Step 8 — Deploy Stage 3: Triton Inference Server 

Now that the model artefacts are in MinIO, deploy Triton:

```bash
kubectl apply -f K8s_Manifests/third_deployment_gpu.yaml
```

Triton will:
1. Pull the GPU-only image `nvcr.io/nvidia/tritonserver:26.02-py3` (this is large — ~8 GB — and may take several minutes to pull (download) on first run).
2. Load `model.onnx` from MinIO automatically on startup.

Monitor the pod until it is ready:

```bash
kubectl get pods -n mlops --watch
# Wait for triton-... to show STATUS = Running and READY = 1/1
```

You can watch Triton's startup logs to confirm the model loaded:

```bash
kubectl logs -n mlops deploy/triton -f
# Look for: "Successfully loaded model 'mathmodel'"
```

> **If the pod crashes:** The most common cause is that the model artefacts were not uploaded correctly in Step 7. Check the logs with `kubectl logs -n mlops deploy/triton`. Also verify the files exist in MinIO at `triton-model-repo/models/mathmodel/config.pbtxt` and `triton-model-repo/models/mathmodel/1/model.onnx`.

---

### Step 9 — Verify Triton Is Serving Predictions

Find the NodePort Triton is listening on:

```bash
kubectl get svc -n mlops triton-http
# Note the NodePort value (e.g. 32175)
```

Get the Minikube IP:

```bash
minikube ip
# e.g. 192.168.49.2
```

Test the health endpoint:

```bash
curl http://$(minikube ip):<PORT>/v2/health/ready
# Expected HTTP 200 with body: {}
```

Test a live inference (student with SATS=105, GCSE=6, AS grade B=4):

```bash
curl -s -X POST \
  http://$(minikube ip):<PORT>/v2/models/mathmodel/versions/1/infer \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test-1",
    "inputs": [{
      "name": "input",
      "shape": [1, 3],
      "datatype": "FP32",
      "data": [[105, 6, 4]]
    }],
    "outputs": [{"name": "output_label"}]
  }'
```

The response will contain `output_label` with an integer 0–6 mapping to grades U/E/D/C/B/A/A*.

---

### Step 10 — Run the k6 Load Test

The load test sends all 1,000 pre-encoded test rows to Triton sequentially and reports latency and error statistics.

Pass the Minikube IP and Triton NodePort as environment variables:

```bash
k6 run \
  -e TRITON_HOST=$(minikube ip) \
  -e TRITON_PORT=<PORT> \
  K6_Test_Scripts/triton_mathmodel_k6_test.js
```

**Expected results:**

| Threshold | Target |
|-----------|--------|
| HTTP error rate | < 1% |
| 95th-percentile latency | < 500 ms |
| Triton inference OK rate | > 99% |
| 99th-percentile latency | < 1,000 ms |

A successful run ends with `✓` checks next to all thresholds.

---

## Grade Reference

| Numeric | Letter | Meaning |
|---------|--------|---------|
| 0 | U | Ungraded |
| 1 | E | E grade |
| 2 | D | D grade |
| 3 | C | C grade (pass threshold) |
| 4 | B | B grade |
| 5 | A | A grade |
| 6 | A* | A* grade |

Pass = grade C or above (numeric ≥ 3).  
Features fed to the model: `[SATS_score (int), GCSE_grade_num (1–9), GCE_AS_grade_num (0–5)]`.

---

## Troubleshooting

**GPU not visible to Kubernetes**
```bash
kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"
# If GPU column shows <none>, ensure:
# 1. nvidia-container-toolkit is installed and Docker is configured to use it
# 2. Minikube was started with --gpus all
# 3. The NVIDIA device plugin DaemonSet is running:
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds
```

**MinIO pod not starting**
```bash
kubectl describe pod -n mlops -l app=minio
# Check for PVC binding errors — you may need to set storageClassName in first_deployment.yaml
```

**Triton crash-loops with "model not found"**
- Confirm the model artefacts were uploaded correctly by the notebook in **Step 6**, and verify the files exist in MinIO at the correct paths.
- Check that `second_deployment.yaml` job completed before running the notebook.

**k6 shows high error rate**
- Confirm Triton is ready (`/v2/health/ready` returns 200) before running the test.
- Double-check the `TRITON_PORT` value from `kubectl get svc -n mlops triton-http`.

**Notebook cannot connect to MinIO**
- The notebook uses the cluster-internal DNS name `minio-api.mlops.svc.cluster.local:9000`. This only works when the notebook is running *inside* the cluster (i.e. in JupyterLab deployed in **Step 2**). Do not run the notebook locally on your machine.


---

## Teardown

To stop and remove all resources:

```bash
kubectl delete namespace mlops
minikube stop
```

To fully delete the Minikube cluster:

```bash
minikube delete
```
