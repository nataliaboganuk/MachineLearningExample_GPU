# UK Student Attainment Predictor — GPU MLOps Pipeline

A full end-to-end machine learning pipeline that trains a Logistic Regression model to predict a student's A-level Maths grade from prior attainment (SATS score, GCSE grade, AS-level grade), then serves live GPU-accelerated predictions via **NVIDIA Triton Inference Server** on a local Kubernetes cluster.

> **This is the GPU version.** Compared to the CPU version, it requires a physical NVIDIA GPU in the host machine, NVIDIA drivers, the NVIDIA Container Toolkit, the NVIDIA Device Plugin for Kubernetes, and a different Triton image and `config.pbtxt`. All GPU-specific steps are clearly marked with a **`[GPU]`** tag throughout this document.

---

## How This Project Works (Big Picture)

```
CSV Dataset (MinIO)
       ↓
JupyterLab Notebook
  • EDA & encoding
  • Train Logistic Regression
  • Export → model.onnx + config.pbtxt  ← KIND_GPU, gpus:[0]
  • Upload artefacts back to MinIO
       ↓
Triton Inference Server (GPU image)
  • Loads model.onnx from MinIO on startup
  • Runs ONNX Runtime on GPU 0
  • Serves predictions over HTTP on port 8000
       ↓
k6 Load Test
  • Fires 1,000 inference requests
  • Measures latency + error rates
```

---

## System Requirements

### Hardware — **[GPU] Required**

| Requirement | Details |
|-------------|---------|
| **NVIDIA GPU** | Any CUDA-capable NVIDIA GPU (Maxwell architecture or newer). The Triton image `26.02-py3` requires CUDA 12.x. |
| RAM | Minimum **12 GB** system RAM (MinIO 2 GB + JupyterLab 2 GB + Triton 4 GB + OS overhead) |
| Disk | Minimum **40 GB** free (Triton GPU image alone is ~15 GB) |

> If you do not have a physical NVIDIA GPU in the machine, use the **CPU version** of this project instead (`third_deployment.yaml` with `tritonserver:26.02-py3` replaced by `tritonserver:26.02-py3-cpu` and `KIND_CPU` in config.pbtxt).

---

### Software to Install

Install all of the following **before** starting the cluster.

| Tool | Version | Purpose |
|------|---------|---------|
| [NVIDIA GPU Driver](https://www.nvidia.com/drivers) | 535+ (CUDA 12.x compatible) | Host GPU driver — **[GPU] required** |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | Latest | Lets Docker/containerd pass the GPU into containers — **[GPU] required** |
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) / Docker Engine | Latest | Container runtime |
| [Minikube](https://minikube.sigs.k8s.io/docs/start/) | v1.32+ | Local Kubernetes cluster |
| [kubectl](https://kubernetes.io/docs/tasks/tools/) | v1.29+ | Kubernetes CLI |
| [k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) | Latest | Load testing tool |
| [NGC Account](https://ngc.nvidia.com) | — | Required to pull the Triton image from `nvcr.io` |

---

## Project Structure

```
GPU_Project_Repo/
├── Data/
│   └── synthetic_uk_attainment_10000_clean_1.csv   # 10,000-row synthetic dataset
├── Notebooks/
│   └── FIXED_Completed_Project.ipynb               # Training & export notebook
├── pbtxt/
│   └── config.pbtxt                                # [GPU] Pre-written GPU Triton config (reference)
├── K8s_Manifests/
│   ├── first_deployment.yaml        # Stage 1 — Namespace, MinIO, JupyterLab
│   ├── second_deployment.yaml       # Stage 2 — Bucket init job
│   ├── third_deployment_gpu.yaml    # [GPU] Stage 3 — Triton with GPU resource requests
│   └── nvidia-device-plugin.yml     # [GPU] REQUIRED — exposes GPU to Kubernetes pods
└── K6_Test_Scripts/
    └── triton_mathmodel_k6_test.js  # Load test (1,000 inference requests)
```

---

## Key Differences vs the CPU Version

| Area | CPU Version | GPU Version |
|------|------------|------------|
| Triton image | `tritonserver:26.02-py3` (cpu variant) | `tritonserver:26.02-py3` (full GPU image) |
| Kubernetes GPU resource | Not requested | `nvidia.com/gpu: 1` in requests **and** limits |
| config.pbtxt instance_group | `kind: KIND_CPU` | `kind: KIND_GPU` + `gpus: [ 0 ]` |
| ONNX batch config | `default-max-batch-size=0` | `default-max-batch-size=8` |
| GPU metrics flag | Not set | `--allow-gpu-metrics=false` |
| NVIDIA Device Plugin | Optional | **Mandatory** — must be applied before Triton |
| Host prerequisites | Docker + Minikube | NVIDIA driver + Container Toolkit + Docker + Minikube |
| Minikube start flags | `--cpus=4 --memory=10240` | `--gpus=all` added |
| Disk space needed | ~10 GB | ~40 GB (GPU image is larger) |

---

## Step-by-Step Instructions

### Step 1 — **[GPU]** Verify Your NVIDIA Driver and GPU

On the host machine (not inside any container), confirm NVIDIA drivers are installed and the GPU is visible:

```bash
nvidia-smi
```

You should see a table showing your GPU name, driver version (535+), and CUDA version (12.x). If this command is not found, install the NVIDIA driver for your operating system from [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers) before continuing.

---

### Step 2 — **[GPU]** Install and Configure the NVIDIA Container Toolkit

The NVIDIA Container Toolkit allows Docker and containerd to pass the GPU through into containers. Without it, Minikube cannot expose the GPU to Kubernetes pods.

**On Ubuntu/Debian:**

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure the Docker runtime to use NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify the toolkit is working:**

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

If the GPU table appears inside the container, the toolkit is correctly installed. If you see an error, do not proceed — Minikube will not be able to access the GPU.

---

### Step 3 — **[GPU]** Start Minikube with GPU Passthrough

The `--gpus=all` flag is what makes the host GPU available inside the Minikube cluster. It requires the Docker driver.

```bash
minikube start \
  --driver=docker \
  --gpus=all \
  --cpus=4 \
  --memory=12288 \
  --disk-size=50g
```

Verify the cluster is running:

```bash
kubectl get nodes
# Expected: one node with STATUS = Ready
```

**[GPU] Confirm the GPU is visible to Kubernetes:**

```bash
kubectl describe node | grep -A5 "Capacity:"
# Look for a line containing: nvidia.com/gpu: 1
```

If `nvidia.com/gpu` does not appear yet, that is expected — it will appear after Step 5 when the NVIDIA Device Plugin is deployed.

---

### Step 4 — Create the NGC Image Pull Secret

Triton is pulled from NVIDIA's private registry (`nvcr.io`) and requires an NGC API key.

1. Log in at [https://ngc.nvidia.com](https://ngc.nvidia.com) and generate a Personal API Key.
2. Run the command below, replacing `YOUR_NGC_API_KEY` with your actual key:

```bash
kubectl create secret docker-registry ngc-registry \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=YOUR_NGC_API_KEY \
  --namespace=mlops
```

> **Note:** The `mlops` namespace does not exist yet — it is created in Step 5. If you get a "namespace not found" error, apply `first_deployment.yaml` first (Step 5), then re-run this command.

---

### Step 5 — Deploy Stage 1: Namespace, MinIO & JupyterLab

This manifest creates the `mlops` namespace, MinIO object storage, JupyterLab, PersistentVolumeClaims, ConfigMaps, Secrets, and RBAC roles.

```bash
kubectl apply -f K8s_Manifests/first_deployment.yaml
```

Wait for both pods to be running:

```bash
kubectl get pods -n mlops --watch
# Wait until minio-... and jupyter-lab-... both show STATUS = Running
```

---

### Step 6 — Deploy Stage 2: Initialise MinIO Buckets

This runs a one-off Kubernetes Job that creates two buckets in MinIO:
- `datasets` — where the CSV dataset will be uploaded
- `triton-model-repo` — where the trained model artefacts will be stored

```bash
kubectl apply -f K8s_Manifests/second_deployment.yaml
```

Confirm the job completed:

```bash
kubectl get jobs -n mlops
# Expected: minio-init-bucket   1/1   Completed
```

If the job shows `0/1` after a couple of minutes, inspect the logs:

```bash
kubectl logs -n mlops job/minio-init-bucket
```

---

### Step 7 — **[GPU]** Deploy the NVIDIA Device Plugin

> **This step is mandatory for the GPU version.** It does not exist as a required step in the CPU project. The NVIDIA Device Plugin is a Kubernetes DaemonSet that advertises the GPU as a schedulable resource (`nvidia.com/gpu`). Without it, the Triton pod will stay in `Pending` state because Kubernetes cannot satisfy the GPU resource request.

```bash
kubectl apply -f K8s_Manifests/nvidia-device-plugin.yml
```

Wait for the plugin pod to be running:

```bash
kubectl get pods -n kube-system --watch | grep nvidia
# Wait for nvidia-device-plugin-daemonset-... STATUS = Running
```

Now confirm the GPU is registered with Kubernetes:

```bash
kubectl describe node | grep -A5 "Allocatable:"
# You must see: nvidia.com/gpu: 1  before proceeding to Step 11
```

If `nvidia.com/gpu` still does not appear, check the plugin logs:

```bash
kubectl logs -n kube-system -l name=nvidia-device-plugin-ds
```

---

### Step 8 — Upload the Dataset to MinIO

Get the MinIO Console URL:

```bash
minikube service minio-webui -n mlops --url
# e.g. http://192.168.49.2:YYYYY
```

Open that URL in your browser and log in with:
- **Username:** `minioadmin`
- **Password:** `minioadminpassword`

Then:
1. Navigate to the **`datasets`** bucket.
2. Click **Upload** and select `Data/synthetic_uk_attainment_10000_clean_1.csv`.
3. Confirm the file appears at the root of the bucket — not inside a subfolder — so the notebook can find it at `datasets/synthetic_uk_attainment_10000_clean_1.csv`.

---

### Step 9 — Open JupyterLab

```bash
minikube service jupyter-lab -n mlops --url
# e.g. http://192.168.49.2:30001
```

Open that URL in a browser. No password is required.

---

### Step 10 — Upload and Run the Notebook

1. In JupyterLab, click the **Upload** button and upload `Notebooks/FIXED_Completed_Project.ipynb`.
2. Open the notebook.
3. Run all cells **in order** from top to bottom using **Run → Run All Cells**.

**What each section of the notebook does:**

| Notebook Section | What It Does |
|-----------------|-------------|
| **Install packages** | Installs pandas, scikit-learn, skl2onnx, onnxruntime, s3fs, etc. |
| **Connect to MinIO** | Connects to `minio-api.mlops.svc.cluster.local:9000` and loads the CSV |
| **Grade mappings** | Defines the 0–6 ordinal scale (U=0, E=1, D=2, C=3, B=4, A=5, A*=6) |
| **EDA** | Prints distributions and renders a pairplot of all attainment variables |
| **Encode & split** | Encodes letter grades to numbers, splits 80/20 train/test with stratification |
| **Train model** | Fits a `StandardScaler → LogisticRegression(lbfgs)` sklearn Pipeline |
| **Evaluate** | Prints accuracy, classification report, and confusion matrices (7-class + pass/fail) |
| **Save .pkl** | Saves the sklearn pipeline to `model.pkl` |
| **Convert to ONNX** | Converts `model.pkl` → `model.onnx`, renames output tensor to `output_label` |
| **Validate ONNX** | Runs a local test inference with `onnxruntime` |
| **[GPU] Generate config.pbtxt** | Writes the Triton config with `KIND_GPU` and `gpus: [0]` — different from CPU version |
| **Upload to MinIO** | Uploads `config.pbtxt` and `model.onnx` to `triton-model-repo/models/mathmodel/` |

> **[GPU] Important — the config.pbtxt generated by the notebook is the GPU version.** It contains `kind: KIND_GPU` and `gpus: [0]`. A pre-written copy of this same file is also available in `pbtxt/config.pbtxt` for reference. Do not substitute the CPU `KIND_CPU` version here.

After the notebook completes successfully you should see:

```
✓ All artefacts uploaded to MinIO successfully.
  triton-model-repo/models/mathmodel/config.pbtxt  (... bytes)
  triton-model-repo/models/mathmodel/1/model.onnx  (... bytes)
```

---

### Step 11 — **[GPU]** Deploy Stage 3: Triton Inference Server (GPU)

> **Use `third_deployment_gpu.yaml` — not `third_deployment.yaml`.** The GPU manifest requests `nvidia.com/gpu: 1`, uses the full Triton image (not the `-cpu` variant), and sets `--allow-gpu-metrics=false` and `--backend-config=onnxruntime,default-max-batch-size=8`.

```bash
kubectl apply -f K8s_Manifests/third_deployment_gpu.yaml
```

Triton will:
1. Pull `nvcr.io/nvidia/tritonserver:26.02-py3` — this is a large image (~15 GB) and will take several minutes on the first run.
2. Initialise the CUDA runtime against GPU 0.
3. Load `model.onnx` from MinIO and place it on the GPU automatically.

Monitor the pod:

```bash
kubectl get pods -n mlops --watch
# Wait for triton-... to show STATUS = Running and READY = 1/1
```

Watch the startup logs to confirm GPU initialisation and model loading:

```bash
kubectl logs -n mlops deploy/triton -f
# Look for lines similar to:
#   "Triton server has ... GPU(s)"
#   "Successfully loaded model 'mathmodel'"
```

> **If the pod stays in `Pending`:** The GPU resource cannot be scheduled. Run `kubectl describe pod -n mlops -l app=triton` and look for `Insufficient nvidia.com/gpu`. This means the NVIDIA Device Plugin from Step 7 has not registered the GPU yet, or the `nvidia-device-plugin.yml` was not applied.

> **If the pod crashes with CUDA errors:** Confirm the host NVIDIA driver is version 535+ and that `nvidia-smi` works correctly on the host. The Triton 26.02 image requires CUDA 12.x.

---

### Step 12 — Verify Triton Is Serving GPU Predictions

Find the Triton HTTP NodePort:

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
curl http://$(minikube ip):<NODEPORT>/v2/health/ready
# Expected: HTTP 200 with body {}
```

**[GPU] Confirm the model is using the GPU:**

```bash
curl -s http://$(minikube ip):<NODEPORT>/v2/models/mathmodel/config | python3 -m json.tool | grep -A5 instance_group
# Look for: "kind": "KIND_GPU"
```

Test a live inference (SATS=105, GCSE grade=6, AS grade B=4):

```bash
curl -s -X POST \
  http://$(minikube ip):<NODEPORT>/v2/models/mathmodel/versions/1/infer \
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

### Step 13 — Run the k6 Load Test

```bash
k6 run \
  -e TRITON_HOST=$(minikube ip) \
  -e TRITON_PORT=<NODEPORT> \
  K6_Test_Scripts/triton_mathmodel_k6_test.js
```

**Expected results:**

| Threshold | Target |
|-----------|--------|
| HTTP error rate | < 1% |
| 95th-percentile latency | < 500 ms |
| Triton inference OK rate | > 99% |
| 99th-percentile latency | < 1,000 ms |

Because the model runs on the GPU, you should see lower latency figures than the CPU version, particularly under concurrent load.

---

## The `pbtxt/config.pbtxt` File Explained

This file is an extra reference included in the GPU project that does not exist in the CPU version. It contains the complete pre-written Triton configuration for GPU deployment:

```protobuf
name: "mathmodel"
backend: "onnxruntime"
max_batch_size: 0

instance_group [
  {
    count: 1
    kind: KIND_GPU      ← runs on the GPU, not CPU
    gpus: [ 0 ]         ← specifically GPU index 0
  }
]
```

The notebook generates and uploads this same content automatically. You only need to use this file manually if the notebook's config generation cell fails and you want to upload `config.pbtxt` directly via the MinIO web console. In that case, upload it to `triton-model-repo/models/mathmodel/config.pbtxt`.

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

**`nvidia-smi` not found on the host**
Install the NVIDIA driver for your OS. On Ubuntu: `sudo apt install nvidia-driver-535`. Reboot after installation.

**`docker run --gpus all` fails**
The NVIDIA Container Toolkit is not installed or `nvidia-ctk runtime configure --runtime=docker` was not run. Repeat Step 2 fully and restart Docker.

**Minikube fails to start with `--gpus=all`**
Ensure you are using `--driver=docker`. The virtualbox and hyperkit drivers do not support GPU passthrough.

**Triton pod stuck in `Pending` — "Insufficient nvidia.com/gpu"**
The NVIDIA Device Plugin has not registered the GPU. Confirm Step 7 was completed and the plugin pod is Running. Check plugin logs: `kubectl logs -n kube-system -l name=nvidia-device-plugin-ds`.

**Triton pod crashes — CUDA driver/library mismatch**
The host driver must be compatible with CUDA 12.x. Run `nvidia-smi` and confirm the CUDA version shown is 12.0 or higher. If not, upgrade the NVIDIA driver.

**Triton pod crashes — "model not found" or "failed to load"**
Confirm the artefacts were uploaded correctly in Step 10. Verify they exist in MinIO at exactly these paths:
- `triton-model-repo/models/mathmodel/config.pbtxt`
- `triton-model-repo/models/mathmodel/1/model.onnx`

Also confirm the `config.pbtxt` contains `KIND_GPU` — if it contains `KIND_CPU`, the notebook used the wrong template.

**Notebook cannot connect to MinIO**
The notebook uses the cluster-internal DNS name `minio-api.mlops.svc.cluster.local:9000`. It must run inside the cluster (in the JupyterLab pod from Step 5). Running the notebook on your local laptop will not work.

**k6 high error rate after deployment**
Wait a full 2–3 minutes after the Triton pod shows `Running` before starting the test — GPU model loading can take additional time after the pod reports ready.

---

## Teardown

To stop and remove all resources:

```bash
kubectl delete namespace mlops
kubectl delete -f K8s_Manifests/nvidia-device-plugin.yml
minikube stop
```

To fully delete the Minikube cluster:

```bash
minikube delete
```
