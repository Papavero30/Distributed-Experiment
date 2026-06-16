# Eksperimen Distributed Volume-Level (tanpa BrainNav)

Harness ringan untuk eksperimen distributed system IEEE: mengukur **throughput,
latency, speedup, efficiency, dan communication overhead** segmentasi aneurisma
volume-level (nnU-Net 2D) pada 3 skala cluster — **tanpa** aplikasi BrainNav.

```
S1 = Laptop saja (RTX 3050Ti)
S2 = Laptop + PC-A (RTX 5080)
S3 = Laptop + PC-A + PC-B (RTX 5080 x2)
```

Worker selalu termasuk laptop (master) sesuai topologi paper. Beban kerja =
volume TOF `10070B` direplikasi **30 task** per konfigurasi.

> ✅ **Mode payload `reference` (default)**: message hanya membawa `scan_id` (kecil).
> Worker membaca file lokal `SCAN_DIR/<scan_id>.nii.gz`. **Wajib copy scan ke tiap
> node sekali** (lihat setup). Eksperimen mengukur **speedup komputasi murni**
> (broker ringan, tidak tercampur waktu transfer). Inilah pilihan yang tepat untuk
> klaim speedup paper.
>
> Alternatif `PAYLOAD_MODE=base64`: volume di-embed di message (~120 MB/task) —
> hanya jika ingin sengaja mengukur *communication overhead*. `queue_wait_ms`
> (transfer+antri) vs `inference_ms` (komputasi) tetap dipisah di kedua mode.

---

## Komponen

| File | Jalan di | Fungsi |
|---|---|---|
| `volume_worker.py` | Laptop, PC-A, PC-B | Consume task → inferensi volume penuh → tulis hasil+timing ke Redis → `/metrics` |
| `publish_tasks.py` | Laptop | Enqueue 30 task volume ke RabbitMQ |
| `collect_results.py` | Laptop | Tunggu selesai → CSV + agregat (throughput/latency/speedup) |

---

## Prasyarat (sekali per mesin)

Tiap node butuh: **venv torch CUDA + nnunetv2**, **folder model `nnUnet-Papavero`**,
dan (untuk worker) file script ini. Laptop sudah punya `nnunet_gpu_env`.

### A. MASTER (laptop) — pastikan stack Docker jalan (PowerShell)
Nyalakan **Docker Desktop** dulu (tunggu status running), lalu:
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\Master-Segmentation"
docker compose up -d rabbitmq redis
# opsional (live dashboard) tanpa menyeret backend:
docker compose up -d --no-deps prometheus grafana
docker ps   # brainnav-rabbitmq(5674), brainnav-redis(6381) harus "Up"
```

---

## RUNBOOK PER NODE

### 1) Setup PC-A & PC-B (via RDP) — sekali saja

> Lakukan untuk PC-A, ulangi untuk PC-B (ganti identitas worker).

1. **Connect**: nyalakan VPN kampus → RDP ke PC-A (LAN `10.28.83.228`) / PC-B (`10.28.83.215`).
2. **Verifikasi bisa lihat master via Tailscale tunnel** (PowerShell di PC):
   ```powershell
   Test-NetConnection 100.79.202.62 -Port 5674   # RabbitMQ → harus TcpTestSucceeded: True
   Test-NetConnection 100.79.202.62 -Port 6381   # Redis
   ```
   Jika `False`: cek tunnel/Tailscale Anda dulu sebelum lanjut.
3. **Salin ke PC** (lewat RDP clipboard/shared drive):
   - folder `distributed-experiment/` dan folder model `nnUnet-Papavero/` (taruh berdampingan, mis. `D:\dist\`)
   - **scan** ke `distributed-experiment\scans\10070B.nii.gz`
     (mode `reference` membaca `SCAN_DIR/<scan_id>.nii.gz`). Sumber: `10070B/10070B/pre/TOF.nii.gz`.
   ```powershell
   mkdir D:\dist\distributed-experiment\scans
   copy <sumber>\TOF.nii.gz D:\dist\distributed-experiment\scans\10070B.nii.gz
   ```
4. **Buat venv GPU** (Python 3.9/3.10/3.11; PowerShell):
   ```powershell
   cd D:\dist\distributed-experiment
   py -3.11 -m venv gpu_env    # atau python -m venv gpu_env
   .\gpu_env\Scripts\python -m pip install --upgrade pip
   .\gpu_env\Scripts\pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   .\gpu_env\Scripts\pip install nnunetv2==2.5.2 --no-deps
   .\gpu_env\Scripts\pip install "numpy==1.26.4" acvl-utils batchgenerators batchgeneratorsv2 `
       "dynamic-network-architectures==0.4.3" einops graphviz imagecodecs matplotlib nibabel `
       pandas requests scikit-image scikit-learn scipy seaborn SimpleITK tifffile tqdm yacs `
       blosc2 msgpack ndindex py-cpuinfo pika redis flask prometheus-client tenacity
   ```
   > Catatan: `nnunetv2 --no-deps` + daftar manual sengaja menghindari `dicom2nifti`
   > yang menarik `python-gdcm` (gagal build di Windows). Sama persis dengan setup laptop.
5. **Verifikasi CUDA**:
   ```powershell
   .\gpu_env\Scripts\python -c "import torch,nnunetv2;print('cuda',torch.cuda.is_available())"
   ```

### 2) Jalankan worker di PC-A
```powershell
cd D:\dist\distributed-experiment
$env:RABBITMQ_HOST="100.79.202.62"; $env:RABBITMQ_PORT="5674"
$env:RABBITMQ_USER="brainnav"; $env:RABBITMQ_PASS="BrainNav_Secure_2025!"
$env:RABBITMQ_VHOST="brainnav_vhost"; $env:VOLUME_QUEUE="segmentation_volume_tasks"
$env:REDIS_URL="redis://:BrainNav_Secure_2025!@100.79.202.62:6381/0"
$env:NNUNET_MODEL_FOLDER="..\nnUnet-Papavero"
$env:PAYLOAD_MODE="reference"; $env:SCAN_DIR=".\scans"   # baca scans\10070B.nii.gz
$env:WORKER_NAME="pc-a-5080"; $env:GPU_TYPE="rtx-5080"; $env:METRICS_PORT="8001"
.\gpu_env\Scripts\python volume_worker.py
```
Worker siap jika muncul: `Worker pc-a-5080 (rtx-5080) menunggu task ...`.
Cek metrics: `http://10.28.83.228:8001/metrics` (atau via Tailscale IP PC-A).

### 3) Worker di PC-B — sama, ganti identitas
```powershell
$env:WORKER_NAME="pc-b-5080"; $env:GPU_TYPE="rtx-5080"; $env:METRICS_PORT="8001"
.\gpu_env\Scripts\python volume_worker.py
```

### 4) Worker baseline di LAPTOP (master) — untuk S1/S2/S3 (PowerShell)
Siapkan scan sekali (laptop juga worker, jadi butuh file lokal):
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\distributed-experiment"
mkdir scans -Force
Copy-Item ..\10070B\10070B\pre\TOF.nii.gz scans\10070B.nii.gz
```
Jalankan worker laptop (gunakan **nnunet_gpu_env**):
```powershell
$env:WORKER_NAME="master-3050ti"; $env:GPU_TYPE="rtx-3050ti-mobile"; $env:METRICS_PORT="8000"
$env:RABBITMQ_HOST="100.79.202.62"; $env:RABBITMQ_PORT="5674"
$env:RABBITMQ_USER="brainnav"; $env:RABBITMQ_PASS="BrainNav_Secure_2025!"; $env:RABBITMQ_VHOST="brainnav_vhost"
$env:REDIS_URL="redis://:BrainNav_Secure_2025!@100.79.202.62:6381/0"
$env:NNUNET_MODEL_FOLDER="..\nnUnet-Papavero"; $env:PAYLOAD_MODE="reference"; $env:SCAN_DIR=".\scans"
..\nnunet_gpu_env\Scripts\python.exe volume_worker.py
```

---

## MENJALANKAN EKSPERIMEN (di laptop)

> Untuk tiap skala: aktifkan worker yang sesuai, lalu publish → collect.
> **Penting:** sebelum tiap run, pastikan queue kosong (purge) agar tidak campur.

Semua perintah di laptop pakai **PowerShell**. Set helper variabel python sekali:
```powershell
cd "D:\MateriKuliahWajib\Tugas Akhir\Implementasi\distributed-experiment"
$PY = "..\nnunet_gpu_env\Scripts\python.exe"
```

Purge queue (opsional, sebelum tiap run; atau via RabbitMQ UI `http://localhost:15674`):
```powershell
& $PY -c "import pika; c=pika.BlockingConnection(pika.ConnectionParameters(host='100.79.202.62',port=5674,virtual_host='brainnav_vhost',credentials=pika.PlainCredentials('brainnav','BrainNav_Secure_2025!'))); c.channel().queue_purge('segmentation_volume_tasks'); print('purged'); c.close()"
```

### S1 — Laptop saja
1. Jalankan **hanya** worker laptop (PC-A/PC-B belum start).
2. Publish + collect (Terminal B):
```powershell
& $PY publish_tasks.py --scan-id 10070B --count 30 --scale S1   # mode reference (default)
# salin run_id yang dicetak, lalu:
& $PY collect_results.py --run-id <RUN_ID_S1> --count 30 --out S1.csv --nodes 1
```

### S2 — Laptop + PC-A
1. Start worker **PC-A** (laptop tetap jalan; PC-B belum).
2.
```powershell
& $PY publish_tasks.py --scan-id 10070B --count 30 --scale S2
& $PY collect_results.py --run-id <RUN_ID_S2> --count 30 --out S2.csv --baseline-csv S1.csv --nodes 2
```

### S3 — Laptop + PC-A + PC-B
1. Start worker **PC-B** (ketiga worker jalan).
2.
```powershell
& $PY publish_tasks.py --scan-id 10070B --count 30 --scale S3
& $PY collect_results.py --run-id <RUN_ID_S3> --count 30 --out S3.csv --baseline-csv S1.csv --nodes 3
```

`collect_results.py` mencetak: wall-clock, throughput, latency (mean/p50/p95),
inference mean, queue-wait mean, distribusi beban per worker, **speedup & efficiency**.

---

## MELIHAT METRICS

### CSV (sumber angka otoritatif untuk paper)
`S1.csv`, `S2.csv`, `S3.csv` — per-task + ringkasan (baris `# wallclock_sec`, `# throughput_per_min`).
Bandingkan ketiganya untuk tabel speedup/efficiency.

### Grafana (monitoring live) — `http://localhost:3005` (admin/admin)
- Datasource Prometheus & dashboard sudah ter-provision
  (`Master-Segmentation/monitoring/grafana/...`).
- Prometheus (`http://localhost:9091`) scrape `/metrics` tiap worker. Pastikan
  target di `Master-Segmentation/monitoring/prometheus.yml` cocok (port 8000/8001;
  ganti IP PC-A/PC-B ke IP yang bisa dijangkau Prometheus jika perlu).
- Metric utama: `volseg_tasks_processed_total`, `volseg_inference_seconds`,
  `volseg_queue_wait_seconds`, `volseg_active_tasks` (label `worker_name`,`gpu_type`).
- Query contoh throughput per worker:
  `rate(volseg_tasks_processed_total[1m])`.

> Jika Prometheus (dalam Docker di laptop) tak bisa menjangkau IP PC-A/PC-B,
> andalkan CSV untuk angka paper; Grafana hanya untuk visual saat run.

---

## Troubleshooting

| Gejala | Penyebab / solusi |
|---|---|
| Worker `AMQPConnectionError` | Tunnel/Tailscale belum jalan; `Test-NetConnection 100.79.202.62 -Port 5674` |
| `message size ... exceeded` | base64 > limit broker; set `max_message_size` di `rabbitmq.conf` (mis. 200MB) atau pakai `PAYLOAD_MODE=reference` |
| Semua task ke 1 worker | Worker lain belum konsumsi / prefetch; pastikan worker lain benar-benar running & subscribe queue sama |
| `collect` timeout | Worker mati di tengah; cek log worker; task ter-requeue (cek RabbitMQ UI) |
| Dice 0 / fg=0 | Wajar untuk subjek sulit; untuk eksperimen distributed yang diukur adalah timing, bukan kualitas |
