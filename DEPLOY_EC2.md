# Deploying SPT-DALES on the AWS EC2 GPU server

Target: `i-0cfe178f5c1ba1f3a` (`3dpointcloud-inference-server`), g6.xlarge, NVIDIA L4
24 GB, us-east-2. Elastic IP `3.17.192.151`.

The five setup scripts in this folder were written for a Lightning Studio but
are host-agnostic — `01_setup_env.sh` detects the GPU's compute capability at
run time (L4 reports 8.9), so no edits are needed. What differs on EC2 is that
the box is long-lived: the service must survive reboots, and the port has to be
opened deliberately.

---

## 0. Copy the scripts up

From your Windows machine, in this folder:

```bash
scp -i <your-key>.pem \
  01_setup_env.sh 02_install_openpcdet.sh 03_install_spt.sh \
  04_verify.py 05_download_checkpoint.sh spt_server.py \
  ubuntu@3.17.192.151:~/
```

If you don't have the `.pem`, use EC2 Instance Connect (the **Connect** button
in the console) and paste the files in, or `git clone` them from a repo.

---

## 1. Build the environment

```bash
ssh -i <your-key>.pem ubuntu@3.17.192.151
bash 01_setup_env.sh        # torch + CUDA stack        (~15 min)
bash 03_install_spt.sh      # superpoint_transformer + FRNN (~25 min)
bash 05_download_checkpoint.sh
python 04_verify.py
```

Skip `02_install_openpcdet.sh` unless you also want PointPillars on this box —
it is only needed for the `.bin`/KITTI path, not for SPT-DALES.

`04_verify.py` must print CUDA available and load the checkpoint. Do not move on
until it does; every failure later in this stack is harder to read than this one.

**The L4 vs. driver note:** `nvidia-smi` reports CUDA 13.2, which is the *driver's*
maximum supported version, not what's installed. The scripts install the cu121
torch build, which runs fine under a newer driver (drivers are backward
compatible). Do not try to match 13.2 — no torch build targets it yet.

---

## 2. Open the port

The server listens on 8000. By default the security group will not allow it.

EC2 console → the instance → **Security** tab → click the security group →
**Edit inbound rules** → **Add rule**:

- Type `Custom TCP`, Port `8000`
- Source: **the JLabel server's IP**, not `0.0.0.0/0`

Restricting the source matters: this endpoint takes an arbitrary URL and
downloads it server-side, so leaving it world-open exposes both the GPU and
whatever your VPC can reach. If QA and this box are in the same VPC, use the
private IP (`172.31.40.3`) and skip public exposure entirely.

---

## 3. Run it as a service

Manual `python spt_server.py` dies with your SSH session. Install a unit:

```bash
sudo tee /etc/systemd/system/spt.service >/dev/null <<'EOF'
[Unit]
Description=SPT-DALES inference server
After=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/miniconda3/envs/spt/bin/python /home/ubuntu/spt_server.py
Restart=always
RestartSec=10
# First inference compiles CUDA kernels and can take minutes; don't let
# systemd's default startup timeout kill it mid-build.
TimeoutStartSec=0
StandardOutput=append:/home/ubuntu/spt.log
StandardError=append:/home/ubuntu/spt.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now spt
sudo systemctl status spt
```

Adjust `ExecStart` to the interpreter `04_verify.py` actually ran under —
`which python` inside the activated env gives it.

Check it answers:

```bash
curl -s localhost:8000/health || curl -s localhost:8000/docs >/dev/null && echo up
tail -f ~/spt.log
```

---

## 4. Point JLabel at it

On the `jtheta_qa` branch, `jlabel/settings/qa.py`:

```python
LIDAR_SPT_INFERENCE_URL = "http://3.17.192.151:8000/predict"
```

(or `http://172.31.40.3:8000/predict` if both boxes share the VPC — faster, and
no public exposure).

```bash
git add jlabel/settings/qa.py
git commit -m "Point QA at the EC2 SPT-DALES inference server"
git push origin jtheta_qa
git checkout lidar_frame
```

**This is the real win over Lightning:** the Elastic IP is static, so this URL
stops going stale. The 4-hour Studio restart cycle that kept breaking QA is gone.

---

## 5. Cost

g6.xlarge on-demand in us-east-2 is roughly **$0.80/hr**, about **$580/month**
if left running continuously. That is far more than the Lightning Pro plan, and
the instance bills whether or not it is doing inference.

If inference is bursty, stop the instance when idle — an Elastic IP survives a
stop, so the URL stays valid. Worth agreeing an on/off policy with your boss
before it runs a full month idle. If it genuinely needs to be always-on, a
1-year Savings Plan cuts it substantially.

---

## Verifying before you trust it

Run one real file end to end rather than assuming the stack is good:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"url": "<a signed S3 URL to a .las/.laz>"}' | head -40
```

The server rejects anything that is not `.las`/`.laz` (`spt_server.py:105`), so
test with an aerial LAS — the file type SPT-DALES was trained on. It has a
3M-point hard cap (`:131`); larger clouds are subsampled.
