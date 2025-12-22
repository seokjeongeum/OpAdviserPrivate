#!/bin/bash
# =============================================================================
# OpAdviser Ultra-Fast Time Evaluation Script
# =============================================================================
# This script runs database tuning and displays the time it took for tuning.
#
# Usage (via Docker):
#   1. Start the container:
#      docker compose up -d
#
#   2. Run the evaluation:
#      docker exec -it OpAdviser sh /root/OpAdviser/eval_time.sh
#
# Or manually inside the container:
#   sh /root/Tuning/eval_time.sh
#   sh /root/OpAdviser/eval_time.sh
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
MAX_RUNS=10          # Number of tuning iterations (reduce for faster demo)
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
    echo "║              Ultra-Fast Time Evaluation Script                    ║"
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
initial_runs = 3
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
    
    # Record start time
    TUNING_START_TIME=$(date +%s)
    echo "${TUNING_START_TIME}" > /tmp/tuning_start_time.txt
    
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
    
    # Record end time
    TUNING_END_TIME=$(date +%s)
    echo "${TUNING_END_TIME}" > /tmp/tuning_end_time.txt
    
    log_info "Tuning completed!"
}

# =============================================================================
# Step 7: Analyze Results - Calculate Tuning Time
# =============================================================================
analyze_results() {
    log_step "Step 7: Analyzing results"
    
    if [ ! -f "${HISTORY_FILE}" ]; then
        log_error "History file not found: ${HISTORY_FILE}"
        exit 1
    fi
    
    # Read start and end times
    if [ -f "/tmp/tuning_start_time.txt" ] && [ -f "/tmp/tuning_end_time.txt" ]; then
        TUNING_START_TIME=$(cat /tmp/tuning_start_time.txt)
        TUNING_END_TIME=$(cat /tmp/tuning_end_time.txt)
        TUNING_DURATION=$((TUNING_END_TIME - TUNING_START_TIME))
    else
        # Fallback: calculate from history file timestamps if available
        TUNING_DURATION=$(python3 << PYEOF
import json
import sys
from datetime import datetime

try:
    with open("${HISTORY_FILE}", 'r') as f:
        data = json.load(f)
    
    history = data.get('data', [])
    if len(history) < 2:
        print("0")
        sys.exit(0)
    
    # Try to get timestamps from history entries
    first_time = None
    last_time = None
    
    for obs in history:
        # Look for timestamp in various possible locations
        timestamp = obs.get('timestamp') or obs.get('time') or obs.get('datetime')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromtimestamp(timestamp)
                
                if first_time is None or dt < first_time:
                    first_time = dt
                if last_time is None or dt > last_time:
                    last_time = dt
            except:
                pass
    
    if first_time and last_time:
        duration = int((last_time - first_time).total_seconds())
        print(duration)
    else:
        print("0")
except Exception as e:
    print("0")
PYEOF
        )
    fi
    
    # Calculate time breakdown
    HOURS=$((TUNING_DURATION / 3600))
    MINUTES=$(((TUNING_DURATION % 3600) / 60))
    SECONDS=$((TUNING_DURATION % 60))
    
    # Get iteration count from history file
    ITERATION_COUNT=$(python3 << PYEOF
import json
try:
    with open("${HISTORY_FILE}", 'r') as f:
        data = json.load(f)
    history = data.get('data', [])
    print(len(history))
except:
    print("0")
PYEOF
    )
    
    # Print results
    echo ""
    echo "================================================================================"
    echo "                    TIME EVALUATION RESULTS"
    echo "================================================================================"
    echo ""
    echo "📊 TUNING SUMMARY"
    echo "--------------------------------------------------------------------------------"
    echo "  Total iterations:     ${ITERATION_COUNT}"
    echo "  Total tuning time:     ${TUNING_DURATION} seconds"
    echo ""
    echo "⏱️  TIME BREAKDOWN"
    echo "--------------------------------------------------------------------------------"
    if [ $HOURS -gt 0 ]; then
        echo "  Total time:            ${HOURS}h ${MINUTES}m ${SECONDS}s"
    elif [ $MINUTES -gt 0 ]; then
        echo "  Total time:            ${MINUTES}m ${SECONDS}s"
    else
        echo "  Total time:            ${SECONDS}s"
    fi
    
    if [ $ITERATION_COUNT -gt 0 ]; then
        AVG_TIME_PER_ITER=$(python3 -c "print(f'{${TUNING_DURATION} / ${ITERATION_COUNT}:.2f}')")
        echo "  Average per iteration: ${AVG_TIME_PER_ITER} seconds"
    fi
    echo ""
    
    # Calculate expected vs actual time
    EXPECTED_TIME=$((MAX_RUNS * (WORKLOAD_TIME + WARMUP_TIME + 10)))
    EFFICIENCY=$(python3 -c "print(f'{(${EXPECTED_TIME} / ${TUNING_DURATION}) * 100:.1f}')" 2>/dev/null || echo "N/A")
    
    echo "📈 TIME ANALYSIS"
    echo "--------------------------------------------------------------------------------"
    echo "  Expected time:          $((EXPECTED_TIME / 60)) minutes ($((EXPECTED_TIME / 3600))h $(((EXPECTED_TIME % 3600) / 60))m)"
    echo "  Start time:            $(date -d @${TUNING_START_TIME} '+%Y-%m-%d %H:%M:%S %Z')"
    echo "  End time:              $(date -d @${TUNING_END_TIME} '+%Y-%m-%d %H:%M:%S %Z')"
    echo "  Actual time:           $((TUNING_DURATION / 60)) minutes ($((TUNING_DURATION / 3600))h $(((TUNING_DURATION % 3600) / 60))m)"
    if [ "$EFFICIENCY" != "N/A" ] && [ $TUNING_DURATION -gt 0 ]; then
        echo "  Time efficiency:       ${EFFICIENCY}%"
    fi
    echo ""
    echo "================================================================================"
    
    # Save summary for external use
    if [ $ITERATION_COUNT -gt 0 ] && [ $TUNING_DURATION -gt 0 ]; then
        AVG_TIME_VAL=$(python3 -c "print(f'{${TUNING_DURATION} / ${ITERATION_COUNT}:.2f}')")
    else
        AVG_TIME_VAL="0.00"
    fi
    
    python3 << PYEOF
import json

summary = {
    "total_iterations": ${ITERATION_COUNT},
    "total_time_seconds": ${TUNING_DURATION},
    "total_time_hours": ${HOURS},
    "total_time_minutes": ${MINUTES},
    "total_time_seconds_remainder": ${SECONDS},
    "average_time_per_iteration": float("${AVG_TIME_VAL}"),
    "expected_time_seconds": ${EXPECTED_TIME},
    "task_id": "${TASK_ID}"
}

with open("/tmp/eval_time_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nDetailed history saved to: ${HISTORY_FILE}")
print(f"Time summary saved to: /tmp/eval_time_summary.json")
print("")
PYEOF
    
    # Print total time in a simple format
    echo ""
    if [ $HOURS -gt 0 ]; then
        echo "Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    elif [ $MINUTES -gt 0 ]; then
        echo "Total Time: ${MINUTES}m ${SECONDS}s"
    else
        echo "Total Time: ${SECONDS}s"
    fi
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
    echo "  - Time summary: /tmp/eval_time_summary.json"
    echo ""
    echo "To view detailed results:"
    echo "  cat /tmp/eval_time_summary.json"
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

