#!/bin/bash
# =============================================================================
# OpAdviser Ultra-Fast Performance Evaluation Script
# =============================================================================
# This script runs database tuning and displays the TPS performance improvement.
#
# Usage (via Docker):
#   1. Start the container:
#      docker compose up -d
#
#   2. Run the evaluation:
#      docker exec -it OpAdviser sh /root/OpAdviser/eval_performance.sh
#
# Or manually inside the container:
#   sh /root/Tuning/eval_performance.sh
#   sh /root/OpAdviser/eval_performance.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
DB_NAME="sbrw"
DB_USER="root"
DB_PASS="password"
DB_HOST="localhost"
DB_PORT="3306"
DB_SOCK="/var/run/mysqld/mysqld.sock"

# Sysbench settings (Ultra-Fast)
TABLES=20
TABLE_SIZE=50000
THREADS=40

# Tuning settings
MAX_RUNS=50          # Number of tuning iterations (reduce for faster demo)
WORKLOAD_TIME=30     # Benchmark duration per iteration (seconds)
WARMUP_TIME=5        # Warmup time per iteration (seconds)

# Paths - auto-detect the correct directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Check common mount locations
if [ -d "/root/Tuning/scripts" ]; then
    OPADVISER_DIR="/root/Tuning"
elif [ -d "/root/OpAdviser/scripts" ]; then
    OPADVISER_DIR="/root/OpAdviser"
elif [ -d "/app/scripts" ]; then
    OPADVISER_DIR="/app"
else
    OPADVISER_DIR="${SCRIPT_DIR}"
fi
CONFIG_FILE="${OPADVISER_DIR}/scripts/config_eval_ultrafast.ini"
TASK_ID="eval_ultrafast_$(date +%Y%m%d_%H%M%S)"
HISTORY_FILE="${OPADVISER_DIR}/repo/history_${TASK_ID}.json"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n"
}

# =============================================================================
# Step 0: Print Banner
# =============================================================================
print_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║   ██████╗ ██████╗  █████╗ ██████╗ ██╗   ██╗██╗███████╗███████╗   ║"
    echo "║  ██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██║   ██║██║██╔════╝██╔════╝   ║"
    echo "║  ██║   ██║██████╔╝███████║██║  ██║██║   ██║██║███████╗█████╗     ║"
    echo "║  ██║   ██║██╔═══╝ ██╔══██║██║  ██║╚██╗ ██╔╝██║╚════██║██╔══╝     ║"
    echo "║  ╚██████╔╝██║     ██║  ██║██████╔╝ ╚████╔╝ ██║███████║███████╗   ║"
    echo "║   ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝╚══════╝╚══════╝   ║"
    echo "║                                                                   ║"
    echo "║           Ultra-Fast Performance Evaluation Script                ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# =============================================================================
# Step 1: Environment Setup
# =============================================================================
setup_environment() {
    log_step "Step 1: Setting up environment"
    
    cd "${OPADVISER_DIR}"
    export PYTHONPATH="${OPADVISER_DIR}"
    export MYSQL_SOCK="${DB_SOCK}"
    
    log_info "Working directory: ${OPADVISER_DIR}"
    log_info "PYTHONPATH: ${PYTHONPATH}"
    log_info "MYSQL_SOCK: ${MYSQL_SOCK}"
}

# =============================================================================
# Step 2: Start MySQL if not running
# =============================================================================
start_mysql() {
    log_step "Step 2: Checking MySQL server"
    
    if pgrep -x "mysqld" > /dev/null; then
        log_info "MySQL is already running"
    else
        log_warn "MySQL is not running, starting..."
        
        # Initialize if needed
        if [ ! -d "/var/lib/mysql/mysql" ]; then
            log_info "Initializing MySQL data directory..."
            mysqld --initialize-insecure --user=mysql
        fi
        
        # Start MySQL
        mysqld --user=mysql &
        
        # Wait for MySQL to be ready
        log_info "Waiting for MySQL to be ready..."
        for i in $(seq 1 30); do
            if mysqladmin ping -h localhost --silent 2>/dev/null; then
                log_info "MySQL is ready!"
                break
            fi
            sleep 1
        done
        
        # Set root password if needed
        if mysql -uroot -e "SELECT 1" 2>/dev/null; then
            log_info "Setting MySQL root password..."
            mysql -uroot -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '${DB_PASS}';"
            mysql -uroot -p${DB_PASS} -e "FLUSH PRIVILEGES;"
        fi
    fi
    
    # Wait for MySQL to be ready
    sleep 5
    log_info "Waiting for MySQL to accept connections..."
    for i in $(seq 1 30); do
        if mysql -u${DB_USER} -p${DB_PASS} -e "SELECT 1" > /dev/null 2>&1; then
            log_info "MySQL is ready!"
            break
        fi
        sleep 1
    done
    # Verify connection
    if mysql -u${DB_USER} -p${DB_PASS} -e "SELECT 1" > /dev/null 2>&1; then
        log_info "MySQL connection verified"
        MYSQL_VERSION=$(mysql -u${DB_USER} -p${DB_PASS} -N -e "SELECT VERSION();")
        log_info "MySQL version: ${MYSQL_VERSION}"
    else
        log_error "Cannot connect to MySQL!"
        exit 1
    fi
}

# =============================================================================
# Step 3: Prepare Database
# =============================================================================
prepare_database() {
    log_step "Step 3: Preparing database"
    
    # Create database
    log_info "Creating database '${DB_NAME}'..."
    mysql -u${DB_USER} -p${DB_PASS} -e "DROP DATABASE IF EXISTS ${DB_NAME}; CREATE DATABASE ${DB_NAME};"
    
    # Prepare sysbench data
    log_info "Preparing sysbench data (${TABLES} tables × ${TABLE_SIZE} rows)..."
    log_info "This may take a few minutes..."
    
    sysbench oltp_read_write \
        --db-driver=mysql \
        --mysql-host=${DB_HOST} \
        --mysql-port=${DB_PORT} \
        --mysql-user=${DB_USER} \
        --mysql-password=${DB_PASS} \
        --mysql-db=${DB_NAME} \
        --tables=${TABLES} \
        --table_size=${TABLE_SIZE} \
        --threads=${THREADS} \
        prepare
    
    log_info "Database prepared successfully!"
}

# =============================================================================
# Step 4: Run Baseline Benchmark
# =============================================================================
run_baseline() {
    log_step "Step 4: Running baseline benchmark (default configuration)"
    
    log_info "Running sysbench with default MySQL settings..."
    
    BASELINE_RESULT=$(sysbench oltp_read_write \
        --db-driver=mysql \
        --mysql-host=${DB_HOST} \
        --mysql-port=${DB_PORT} \
        --mysql-user=${DB_USER} \
        --mysql-password=${DB_PASS} \
        --mysql-db=${DB_NAME} \
        --tables=${TABLES} \
        --table_size=${TABLE_SIZE} \
        --threads=${THREADS} \
        --time=${WORKLOAD_TIME} \
        --report-interval=10 \
        run 2>&1)
    
    # Extract TPS from result
    BASELINE_TPS=$(echo "${BASELINE_RESULT}" | grep "transactions:" | awk '{print $3}' | tr -d '(')
    
    if [ -z "${BASELINE_TPS}" ]; then
        BASELINE_TPS=$(echo "${BASELINE_RESULT}" | grep -oP 'transactions:\s+\d+\s+\(\K[\d.]+')
    fi
    
    log_info "Baseline TPS: ${BASELINE_TPS}"
    echo "${BASELINE_TPS}" > /tmp/baseline_tps.txt
}

# =============================================================================
# Step 5: Create Configuration File
# =============================================================================
create_config() {
    log_step "Step 5: Creating tuning configuration"
    
    cat > "${CONFIG_FILE}" << EOF
[database]
db = mysql
host = ${DB_HOST}
port = ${DB_PORT}
user = ${DB_USER}
passwd = ${DB_PASS}
sock = ${DB_SOCK}
cnf = scripts/template/experiment_normandy.cnf
mysqld = /usr/sbin/mysqld
pg_ctl = /usr/bin/pg_ctl
pgdata = /var/lib/postgresql/data
postgres = /usr/bin/postgres
knob_config_file = scripts/experiment/gen_knobs/mysql_dynamic_10.json
knob_num = 10
dbname = ${DB_NAME}
workload = sysbench
oltpbench_config_xml = 
workload_type = sbrw
thread_num = ${THREADS}
workload_warmup_time = ${WARMUP_TIME}
workload_time = ${WORKLOAD_TIME}
remote_mode = False
ssh_user = 
online_mode = True
isolation_mode = False
pid = 0
lhs_log = 
cpu_core = 

[tune]
task_id = ${TASK_ID}
performance_metric = ['tps']
reference_point = [None, None]
constraints = 
max_runs = ${MAX_RUNS}
optimize_method = SMAC
space_transfer = False
auto_optimizer = False
selector_type = shap
initial_runs = 10
initial_tunable_knob_num = 10
incremental = none
incremental_every = 10
incremental_num = 2
acq_optimizer_type = local_random
batch_size = 8
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

    log_info "Configuration saved to: ${CONFIG_FILE}"
    log_info "Task ID: ${TASK_ID}"
}

# =============================================================================
# Step 6: Run Tuning
# =============================================================================
run_tuning() {
    log_step "Step 6: Running database tuning (${MAX_RUNS} iterations)"
    
    ESTIMATED_TIME=$((MAX_RUNS * (WORKLOAD_TIME + WARMUP_TIME + 10) / 60))
    log_info "Estimated time: ~${ESTIMATED_TIME} minutes"
    log_info "Progress will be shown below..."
    echo ""
    
    # Remove existing history file to start fresh
    rm -f "${HISTORY_FILE}"
    
    # Run the optimization
    python scripts/optimize.py --config="${CONFIG_FILE}" 2>&1 | while IFS= read -r line; do
        # Print progress updates
        if echo "$line" | grep -q "Iteration"; then
            echo -e "${CYAN}$line${NC}"
        elif echo "$line" | grep -q "objective"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -q "Best"; then
            echo -e "${BOLD}$line${NC}"
        elif echo "$line" | grep -q "ERROR\|Error\|error"; then
            echo -e "${RED}$line${NC}"
        fi
    done
    
    log_info "Tuning completed!"
}

# =============================================================================
# Step 7: Analyze Results
# =============================================================================
analyze_results() {
    log_step "Step 7: Analyzing results"
    
    if [ ! -f "${HISTORY_FILE}" ]; then
        log_error "History file not found: ${HISTORY_FILE}"
        exit 1
    fi
    
    # Extract TPS values using Python
    python3 << EOF
import json
import sys

# Load history
with open("${HISTORY_FILE}", 'r') as f:
    data = json.load(f)

history = data.get('data', [])

if not history:
    print("No tuning data found!")
    sys.exit(1)

# Extract TPS values
tps_values = []
for obs in history:
    tps = obs.get('external_metrics', {}).get('tps', 0)
    if tps > 0.1:  # Filter out failed runs
        tps_values.append(tps)

if not tps_values:
    print("No valid TPS measurements found!")
    sys.exit(1)

# Calculate statistics
default_tps = tps_values[0] if tps_values else 0
best_tps = max(tps_values)
worst_tps = min(tps_values)
avg_tps = sum(tps_values) / len(tps_values)

# Practical worst (exclude outliers below 10% of median)
sorted_tps = sorted(tps_values)
median_tps = sorted_tps[len(sorted_tps) // 2]
threshold = median_tps * 0.1
practical_tps = [t for t in tps_values if t >= threshold]
practical_worst = min(practical_tps) if practical_tps else worst_tps

# Calculate improvements
best_vs_worst = best_tps / worst_tps if worst_tps > 0 else float('inf')
best_vs_practical = best_tps / practical_worst if practical_worst > 0 else float('inf')

# Find best iteration
best_iter = tps_values.index(best_tps) + 1

# Print results
print("")
print("=" * 70)
print("               PERFORMANCE EVALUATION RESULTS")
print("=" * 70)
print("")
print(f"📊 TUNING SUMMARY")
print("-" * 70)
print(f"  Total iterations:     {len(history)}")
print(f"  Valid measurements:   {len(tps_values)}")
print(f"  Best iteration:       #{best_iter}")
print("")
print(f"📈 TPS MEASUREMENTS")
print("-" * 70)
print(f"  Default (iter 1):     {default_tps:,.2f} TPS")
print(f"  Worst TPS:            {worst_tps:,.2f} TPS")
print(f"  Practical Worst:      {practical_worst:,.2f} TPS")
print(f"  Best TPS:             {best_tps:,.2f} TPS")
print(f"  Average TPS:          {avg_tps:,.2f} TPS")
print("")
print("=" * 70)
print("        ⚡ PERFORMANCE IMPROVEMENT RATIOS ⚡")
print("=" * 70)
print("")
print(f"  🔸 Best / Worst:           {best_vs_worst:,.2f}x")
print(f"  🔸 Best / Practical Worst: {best_vs_practical:,.2f}x")
print("")

# Target check
TARGET = 4.5
max_improvement = max(best_vs_worst, best_vs_practical)
print("-" * 70)
print(f"  🎯 Target improvement:    {TARGET}x")
print("")

if max_improvement >= TARGET:
    print(f"  ✅ SUCCESS! Maximum improvement: {max_improvement:.2f}x >= {TARGET}x")
else:
    print(f"  ⚠️  Below target: {max_improvement:.2f}x < {TARGET}x")

print("")
print("=" * 70)

# Save summary for external use
summary = {
    "default_tps": default_tps,
    "best_tps": best_tps,
    "worst_tps": worst_tps,
    "practical_worst_tps": practical_worst,
    "best_vs_worst": best_vs_worst,
    "best_vs_practical": best_vs_practical,
    "best_iteration": best_iter,
    "total_iterations": len(history)
}

with open("/tmp/eval_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDetailed history saved to: ${HISTORY_FILE}")
print(f"Summary saved to: /tmp/eval_summary.json")
print("")
EOF
}

# =============================================================================
# Step 8: Print Final Summary
# =============================================================================
print_summary() {
    log_step "Evaluation Complete!"
    
    echo -e "${GREEN}${BOLD}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                    EVALUATION COMPLETED                           ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo ""
    echo "Files generated:"
    echo "  - Tuning history: ${HISTORY_FILE}"
    echo "  - Summary: /tmp/eval_summary.json"
    echo ""
    echo "To view detailed results:"
    echo "  cat /tmp/eval_summary.json"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    print_banner
    setup_environment
    start_mysql
    prepare_database
    run_baseline
    create_config
    run_tuning
    analyze_results
    print_summary
}

# Run main function
main "$@"

