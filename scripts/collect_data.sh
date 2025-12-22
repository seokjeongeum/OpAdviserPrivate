#!/bin/bash
# =============================================================================
# Resource Data Collection Script for CPU & I/O Prediction Model Training
# =============================================================================
# This script collects resource usage data including:
#   - CPU usage (%)
#   - Read I/O (MB/s)
#   - Write I/O (MB/s)
#
# Steps:
# 1. Creates MySQL config with smaller buffer pool to generate I/O activity
# 2. Sets up sysbench dataset that exceeds buffer pool (forces disk I/O)
# 3. Collects resource data with varying knob configurations
# 4. Trains the resource prediction model
# =============================================================================

set -e  # Exit on error

# Configuration
OPADVISER_DIR="/root/OpAdviser"
MYSQL_USER="root"
MYSQL_PASSWORD="password"
MYSQL_DB="sbrw"
MYSQL_HOST="localhost"
MYSQL_PORT="3306"

# Buffer pool size (256MB - small enough to force I/O)
BUFFER_POOL_SIZE="268435456"

# Sysbench settings
NUM_TABLES=5
TABLE_SIZE=500000  # 500K rows per table (~1.25GB total, exceeds 256MB buffer)

# Data collection settings
NUM_SAMPLES=60
OUTPUT_DIR="resource_data"

# Knob configuration
KNOB_CONFIG="scripts/experiment/gen_knobs/mysql_cpu_io_dynamic_15.json"
KNOB_NUM=15

# Model output
MODEL_OUTPUT_DIR="resource_models"

# =============================================================================
# Helper functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# =============================================================================
# Step 1: Create MySQL config with smaller buffer pool
# =============================================================================

log "Step 1: Creating MySQL config with smaller buffer pool..."

cd "$OPADVISER_DIR"

# Create new config file
ORIGINAL_CNF="scripts/template/experiment_normandy.cnf"
RESOURCE_CNF="scripts/template/experiment_resource.cnf"

if [ ! -f "$ORIGINAL_CNF" ]; then
    error "Original config file not found: $ORIGINAL_CNF"
fi

cp "$ORIGINAL_CNF" "$RESOURCE_CNF"
sed -i "s/innodb_buffer_pool_size.*/innodb_buffer_pool_size = $BUFFER_POOL_SIZE/" "$RESOURCE_CNF"

log "  Created $RESOURCE_CNF with buffer_pool_size=$BUFFER_POOL_SIZE (256MB)"

# =============================================================================
# Step 2: Create/update collection config
# =============================================================================

log "Step 2: Setting up collection config..."

COLLECTION_CONFIG="$OUTPUT_DIR/collection_config.ini"
mkdir -p "$OUTPUT_DIR"

cat > "$COLLECTION_CONFIG" << EOF
[database]
db = mysql
host = $MYSQL_HOST
port = $MYSQL_PORT
user = $MYSQL_USER
passwd = $MYSQL_PASSWORD
sock = /var/run/mysqld/mysqld.sock
cnf = $RESOURCE_CNF
mysqld = /usr/sbin/mysqld
pg_ctl = /usr/bin/pg_ctl
pgdata = /var/lib/postgresql/data
postgres = /usr/bin/postgres

# CPU/IO focused: Use dynamic knobs only
knob_config_file = $KNOB_CONFIG
knob_num = $KNOB_NUM

# Workload settings
dbname = $MYSQL_DB
workload = sysbench
oltpbench_config_xml = 
workload_type = sbrw
thread_num = 40

# Benchmark time settings
workload_warmup_time = 5
workload_time = 30

# Mode settings
remote_mode = False
ssh_user = 
online_mode = True
isolation_mode = False
pid = 0
lhs_log = 
cpu_core = 

[tune]
task_id = resource_data_collection

performance_metric = ['tps']
reference_point = [None, None]
constraints = 

# Collection settings
max_runs = $NUM_SAMPLES
optimize_method = SMAC
space_transfer = False
auto_optimizer = False

selector_type = shap
initial_runs = $NUM_SAMPLES
initial_tunable_knob_num = $KNOB_NUM
incremental = none
incremental_every = 10
incremental_num = 2

acq_optimizer_type = random
batch_size = 1
mean_var_file = 
params = 
tr_init = True
replay_memory = 

transfer_framework = none
data_repo = repo

only_knob = False
only_range = False
latent_dim = 0
EOF

log "  Created $COLLECTION_CONFIG"

# =============================================================================
# Step 3: Cleanup and create sysbench tables
# =============================================================================

log "Step 3: Setting up sysbench dataset..."

log "  Cleaning up existing tables..."
sysbench /usr/share/sysbench/oltp_read_write.lua \
    --mysql-host="$MYSQL_HOST" \
    --mysql-port="$MYSQL_PORT" \
    --mysql-user="$MYSQL_USER" \
    --mysql-password="$MYSQL_PASSWORD" \
    --mysql-db="$MYSQL_DB" \
    --tables="$NUM_TABLES" \
    cleanup 2>/dev/null || true

log "  Creating $NUM_TABLES tables with $TABLE_SIZE rows each..."
log "  (This may take several minutes...)"

sysbench /usr/share/sysbench/oltp_read_write.lua \
    --mysql-host="$MYSQL_HOST" \
    --mysql-port="$MYSQL_PORT" \
    --mysql-user="$MYSQL_USER" \
    --mysql-password="$MYSQL_PASSWORD" \
    --mysql-db="$MYSQL_DB" \
    --tables="$NUM_TABLES" \
    --table-size="$TABLE_SIZE" \
    prepare

log "  Sysbench tables created successfully"

# Show table sizes
log "  Table sizes:"
mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" -e \
    "SELECT table_name, ROUND(data_length/1024/1024, 2) AS 'Size (MB)' 
     FROM information_schema.tables 
     WHERE table_schema='$MYSQL_DB' AND table_name LIKE 'sbtest%';" 2>/dev/null

# =============================================================================
# Step 4: Restart MySQL with new buffer pool (if needed)
# =============================================================================

log "Step 4: Checking MySQL buffer pool size..."

CURRENT_BUFFER=$(mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" -N -e \
    "SELECT @@innodb_buffer_pool_size;" 2>/dev/null)

log "  Current buffer pool size: $CURRENT_BUFFER bytes"
log "  Target buffer pool size: $BUFFER_POOL_SIZE bytes"

if [ "$CURRENT_BUFFER" != "$BUFFER_POOL_SIZE" ]; then
    log "  Note: Buffer pool size will be applied when MySQL restarts with new config."
    log "  The data collection script will handle config changes dynamically."
fi

# =============================================================================
# Step 5: Collect resource data (CPU, Read I/O, Write I/O)
# =============================================================================

log "Step 5: Collecting resource data ($NUM_SAMPLES samples)..."
log "  Metrics: CPU usage, Read I/O, Write I/O"
log "  Estimated time: ~$(( NUM_SAMPLES * 40 / 60 )) minutes"

python scripts/collect_resource_data.py \
    --config "$COLLECTION_CONFIG" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR"

# Verify data was collected
if [ ! -f "$OUTPUT_DIR/resource_data.json" ]; then
    error "Data collection failed - no output file found"
fi

log "  Data collection complete!"

# Check resource values in collected data
log "  Verifying collected resource data..."
python3 -c "
import json
d = json.load(open('$OUTPUT_DIR/resource_data.json'))
data = d['data']
print(f'  Samples collected: {len(data)}')

cpus = [s.get('resource', {}).get('cpu', 0) for s in data]
read_ios = [s.get('resource', {}).get('readIO', 0) for s in data]
write_ios = [s.get('resource', {}).get('writeIO', 0) for s in data]

print(f'  CPU      - min: {min(cpus):.2f}%, max: {max(cpus):.2f}%, mean: {sum(cpus)/len(cpus):.2f}%')
print(f'  Read I/O - min: {min(read_ios):.2f}, max: {max(read_ios):.2f}, mean: {sum(read_ios)/len(read_ios):.2f} MB/s')
print(f'  Write I/O - min: {min(write_ios):.2f}, max: {max(write_ios):.2f}, mean: {sum(write_ios)/len(write_ios):.2f} MB/s')

# Warn if I/O is all zeros
if max(read_ios) == 0 and max(write_ios) == 0:
    print('  WARNING: All I/O values are zero! Buffer pool may still be too large.')
"

# =============================================================================
# Step 6: Train resource prediction model
# =============================================================================

log "Step 6: Training resource prediction model..."

python scripts/train_resource_model.py \
    --data_file "$OUTPUT_DIR/resource_data.json" \
    --knob_config "$KNOB_CONFIG" \
    --knob_num "$KNOB_NUM" \
    --output_dir "$MODEL_OUTPUT_DIR"

# =============================================================================
# Summary
# =============================================================================

log "============================================================"
log "Resource Data Collection Complete!"
log "============================================================"
log "Collected metrics: CPU usage, Read I/O, Write I/O"
log "Data saved to: $OUTPUT_DIR/resource_data.json"
log "Model saved to: $MODEL_OUTPUT_DIR/resource_predictor.joblib"
log "Metadata saved to: $MODEL_OUTPUT_DIR/training_metadata.json"
log "============================================================"

# Show final metrics
if [ -f "$MODEL_OUTPUT_DIR/training_metadata.json" ]; then
    log "Training Results:"
    python3 -c "
import json
import math
m = json.load(open('$MODEL_OUTPUT_DIR/training_metadata.json'))
cpu_mape = m['test_metrics']['cpu_mape']
read_io_mape = m['test_metrics']['read_io_mape']
write_io_mape = m['test_metrics']['write_io_mape']
combined_io_mape = m['test_metrics'].get('combined_io_mape')
if combined_io_mape is None:
    # Calculate combined I/O MAPE if not present
    if math.isinf(read_io_mape) and math.isinf(write_io_mape):
        combined_io_mape = float('inf')
    elif math.isinf(read_io_mape):
        combined_io_mape = write_io_mape
    elif math.isinf(write_io_mape):
        combined_io_mape = read_io_mape
    else:
        combined_io_mape = (read_io_mape + write_io_mape) / 2.0

cpu_str = f'{cpu_mape:.2f}' if not math.isinf(cpu_mape) else 'inf'
read_str = f'{read_io_mape:.2f}' if not math.isinf(read_io_mape) else 'inf'
write_str = f'{write_io_mape:.2f}' if not math.isinf(write_io_mape) else 'inf'
combined_str = f'{combined_io_mape:.2f}' if not math.isinf(combined_io_mape) else 'inf'

print(f\"  CPU MAPE:      {cpu_str}%\")
print(f\"  Read I/O MAPE: {read_str}%\")
print(f\"  Write I/O MAPE: {write_str}%\")
print(f\"  I/O MAPE (combined): {combined_str}%\")
print(f\"  All <10% MAPE: {'✓ PASS' if m['test_metrics']['all_pass'] else '✗ FAIL'}\")
"
fi

log "Done!"

