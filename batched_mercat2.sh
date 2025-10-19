#!/usr/bin/env bash
set -euo pipefail

# --- edit these if needed ---
WORK="/scratch/home/glh52/hgt"           # mounted to /work
IN_DIR="${WORK}/split_positive"          # folder with .faa files
FINAL="${WORK}/hgt_negative_2mers_combined.tsv"
BATCH_SIZE=5000
K=2
C=0
N=32
IMAGE="quay.io/biocontainers/mercat2:1.4.1--pyhdfd78af_0"
PY_IMAGE="python:3.11"
# ----------------------------

mkdir -p "${WORK}/batches" "${WORK}/runs" "${WORK}/keep" "${WORK}/tmp"

echo "[1/4] Indexing inputs and making batches of ${BATCH_SIZE}…"
# we only need basenames since everything is in one folder
find "${IN_DIR}" -maxdepth 1 -type f -name '*.faa' -printf '%f\n' | sort > "${WORK}/all_faa.txt"
split -l "${BATCH_SIZE}" "${WORK}/all_faa.txt" "${WORK}/batches/list_"

echo "[2/4] Running MerCat2 per batch (keeping only combined_protein.tsv)…"
i=0
for LIST in $(ls -1 "${WORK}/batches"/list_* | sort); do
  i=$((i+1))
  BATCH_TAG="batch_$(printf "%05d" "${i}")"
  OUT_DIR="/work/runs/${BATCH_TAG}"
  echo "  -> ${BATCH_TAG}"

  # --- REPLACE your old docker run block with this watchdog version ---
  docker run --rm \
    --ulimit nofile=1000000:1000000 \
    -v "${IN_DIR}:/in:ro" \
    -v "${WORK}:/work" \
    -e TMPDIR=/work/tmp \
    "${IMAGE}" bash -lc "
      set -euo pipefail
      BATCH_TAG='${BATCH_TAG}'
      OUT_DIR='${OUT_DIR}'

      # Build -i list for just this batch
      FILES=\$(awk '{printf \"/in/%s \", \$0}' /work/batches/$(basename "${LIST}"))
      mkdir -p \"\$OUT_DIR\"

      # Start MerCat2 in background
      ( mercat2.py -i \$FILES -k ${K} -c ${C} -n ${N} -o \"\$OUT_DIR\" -replace ) &
      pid=\$!

      target=\"\$OUT_DIR/combined_protein.tsv\"
      stable=0
      prev_size=0
      waited=0
      SLEEP=5           # seconds between checks
      STABLE_ROUNDS=3   # ~15s with no size change
      TIMEOUT=3600      # max seconds to wait per batch

      # Wait for TSV to appear and become stable, then stop MerCat2
      while kill -0 \$pid 2>/dev/null; do
        if [ -f \"\$target\" ]; then
          size=\$(wc -c < \"\$target\" 2>/dev/null || echo 0)
          if [ \"\$size\" -gt 0 ] && [ \"\$size\" = \"\$prev_size\" ]; then
            stable=\$((stable+1))
          else
            stable=0
          fi
          prev_size=\$size
          if [ \$stable -ge \$STABLE_ROUNDS ]; then
            echo \"[watchdog] TSV is stable; stopping MerCat2 (pid \$pid)…\"
            kill -INT \$pid 2>/dev/null || true
            sleep 5
            kill -TERM \$pid 2>/dev/null || true
            break
          fi
        fi
        sleep \$SLEEP
        waited=\$((waited+SLEEP))
        if [ \$waited -ge \$TIMEOUT ]; then
          echo \"[watchdog] Timeout reached; stopping MerCat2 (pid \$pid)…\" >&2
          kill -INT \$pid 2>/dev/null || true
          sleep 5
          kill -TERM \$pid 2>/dev/null || true
          break
        fi
      done

      # Don't fail if we killed the job; finish cleanup
      wait \$pid 2>/dev/null || true

      # Keep the per-batch matrix and remove heavy extras
      mkdir -p /work/keep
      cp \"\$target\" \"/work/keep/${BATCH_TAG}.tsv\"
      rm -rf \"\$OUT_DIR/report\" \"\$OUT_DIR/tsv_protein\" \"\$OUT_DIR/combined_protein_T.tsv\" || true
    "
  # --- END replacement block ---
done


echo "[3/4] Merging all batch matrices into one (outer-join on k-mer)…"
docker run --rm -v "${WORK}:/work" python:3.11-slim bash -lc '
set -euo pipefail
pip install --no-cache-dir -q pandas
python - << "PY"
import os, glob
import pandas as pd

keep = "/work/keep"
out  = "/work/hgt_negative_2mers_combined.tsv"

files = sorted(glob.glob(os.path.join(keep, "*.tsv")))
if not files:
    raise SystemExit("No batch TSVs found in /work/keep")

merged = None
key = None
for f in files:
    t = pd.read_csv(f, sep="\t")
    if key is None:
        key = t.columns[0]        # k-mer column name
    elif t.columns[0] != key:
        t = t.rename(columns={t.columns[0]: key})
    merged = t if merged is None else pd.merge(merged, t, on=key, how="outer")

merged = merged.fillna(0).sort_values(by=[key]).reset_index(drop=True)
merged.to_csv(out, sep="\t", index=False)
print(f"Wrote {out} with shape {merged.shape}")
PY
'

echo "[4/4] Deleting temp content inside Docker (leave only final file)…"
docker run --rm -v "${WORK}:/work" "${PY_IMAGE}" bash -lc "
  set -euo pipefail
  rm -rf /work/runs /work/keep /work/batches /work/tmp /work/all_faa.txt
  ls -lh /work | grep '$(basename "${FINAL}")' || (echo 'Final file missing!' >&2; exit 1)
"

echo "Done ✅  Final matrix at: ${FINAL}"

