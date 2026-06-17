#!/usr/bin/env bash
# End-to-end GCP GPU run for the phase 5b change probe: provision a T4 VM, run the same
# platform-agnostic runner as Kaggle/Colab (kaggle/run_change_probe.py), pull the results, then
# DELETE the VM. The VM is torn down on ANY exit (success, failure, Ctrl+C) via a trap, so a
# forgotten GPU instance can't quietly burn the trial credit.
#
# Prereqs (see gcp/README.md): an activated (paid) GCP account with GPU quota >= 1, the `gcloud`
# CLI authenticated (`gcloud auth login` + `gcloud config set project <id>`), the Compute Engine
# API enabled, and a GitHub PAT in $GH_PAT.
#
# Usage:
#   export GH_PAT=ghp_xxx
#   bash gcp/run_change_probe.sh            # uses defaults below
#   ZONE=europe-west4-a GPU=nvidia-tesla-t4 bash gcp/run_change_probe.sh
set -euo pipefail

# --- config (override via env) ------------------------------------------------------------------
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-europe-west4-a}"          # T4 availability + EU region
MACHINE="${MACHINE:-n1-standard-4}"
GPU="${GPU:-nvidia-tesla-t4}"
VM="${VM:-change-probe-$(date +%s 2>/dev/null || echo run)}"
SPOT="${SPOT:-1}"                       # 1 = cheap preemptible Spot VM; ok for a short job
OUT_DIR="${OUT_DIR:-gcp/_out}"
GH_REPO="github.com/AstroCan17/geo-embed-eo-cdk.git"
BRANCH="${BRANCH:-main}"                 # which branch the VM clones (override to test a PR branch)
# Deep Learning VM: CUDA driver + PyTorch preinstalled, so no driver wrangling. Google renames
# these families over time (the old `pytorch-latest-gpu` is gone); pin a current one and override
# via $IMG_FAMILY if it ages out (list with: gcloud compute images list --project deeplearning-platform-release).
IMG_FAMILY="${IMG_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMG_PROJECT="deeplearning-platform-release"

[ -n "${GH_PAT:-}" ] || { echo "ERROR: set GH_PAT (a GitHub PAT with repo read access)"; exit 1; }
[ -n "$PROJECT" ] || { echo "ERROR: no project — run 'gcloud config set project <id>'"; exit 1; }

spot_flags=()
[ "$SPOT" = "1" ] && spot_flags=(--provisioning-model=SPOT --instance-termination-action=DELETE)

cleanup() {
  echo "+ deleting VM $VM (cleanup) ..."
  gcloud compute instances delete "$VM" --zone "$ZONE" --project "$PROJECT" --quiet 2>/dev/null || true
}
trap cleanup EXIT

# --- 1) provision -------------------------------------------------------------------------------
echo "+ creating GPU VM $VM ($GPU, $MACHINE, $ZONE, spot=$SPOT) ..."
gcloud compute instances create "$VM" \
  --project "$PROJECT" --zone "$ZONE" \
  --machine-type "$MACHINE" \
  --accelerator "type=$GPU,count=1" \
  --image-family "$IMG_FAMILY" --image-project "$IMG_PROJECT" \
  --maintenance-policy TERMINATE \
  --metadata install-nvidia-driver=True \
  --boot-disk-size 100GB \
  "${spot_flags[@]}"

# wait until SSH is reachable (driver install on first boot can take a couple of minutes)
echo "+ waiting for SSH ..."
for i in $(seq 1 30); do
  gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --command "true" 2>/dev/null && break
  sleep 10
done

# --- 2) hand the token over out-of-band (never on the command line) -----------------------------
echo "+ transferring PAT (hidden) ..."
printf '%s' "$GH_PAT" | gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" \
  --command 'umask 077; cat > "$HOME/.ghpat"'

# --- 3) run the same portable runner ------------------------------------------------------------
echo "+ running change probe on GPU ..."
gcloud compute ssh "$VM" --zone "$ZONE" --project "$PROJECT" --command '
  set -e
  export GH_PAT="$(cat "$HOME/.ghpat")"; rm -f "$HOME/.ghpat"
  export GEO_WORK="$HOME/work" GEO_REPO="$HOME/work/repo"
  mkdir -p "$GEO_WORK"
  git clone --depth 1 -b '"$BRANCH"' "https://x-access-token:${GH_PAT}@'"$GH_REPO"'" "$GEO_REPO" 2>/dev/null
  # On this DL VM torch lives in the system dist-packages (root-owned, not writable by the login
  # user), so a plain `pip install` silently lands in ~/.local and the runner can not import it.
  # Run the runner as root so its pip install goes system-wide, next to torch. (conda is absent here;
  # the conda branch is kept for other DL images.)
  PY=/opt/conda/bin/python; [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
  sudo --preserve-env=GH_PAT,GEO_WORK,GEO_REPO "$PY" "$GEO_REPO/kaggle/run_change_probe.py"
  sudo chmod -R a+rX "$GEO_WORK"
'

# --- 4) fetch results ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
echo "+ fetching results to $OUT_DIR/ ..."
gcloud compute scp "$VM":'~/work/change_probe_results.md' "$OUT_DIR/" \
  --zone "$ZONE" --project "$PROJECT"

echo "+ done — results at $OUT_DIR/change_probe_results.md (VM will be deleted on exit)"
