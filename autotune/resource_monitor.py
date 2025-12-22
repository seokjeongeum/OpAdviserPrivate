import pdb
import time
import logging
import psutil
import subprocess
import multiprocessing as mp
from multiprocessing import Manager

logger = logging.getLogger("autotune.resource_monitor")


class ResourceMonitor:

    def __init__(self, pid, interval, warmup, t):
        self.interval = interval
        self.t = t
        self.process = psutil.Process(pid)
        self.warmup = warmup
        self.ticks = int(self.t / self.interval)
        self.n_cpu = len(self.process.cpu_affinity())
        self.cpu_usage_seq = Manager().list()
        self.mem_virtual_usage_seq = Manager().list()
        self.mem_physical_usage_seq = Manager().list()
        self.io_read_seq, self.io_write_seq = Manager().list(), Manager().list()
        self.dirty_pages_pct_seq = Manager().list()
        self.processes = []
        self.alive = mp.Value('b', False)

    def run(self):
        p1 = mp.Process(target=self.monitor_cpu_usage, args=())
        self.processes.append(p1)
        p2 = mp.Process(target=self.monitor_mem_usage, args=())
        self.processes.append(p2)
        p3 = mp.Process(target=self.monitor_io_usage, args=())
        self.processes.append(p3)
        self.alive.value = True
        [proc.start() for proc in self.processes]

    def get_monitor_data(self):
        [proc.join() for proc in self.processes]
        return {
            'mem_virtual': list(self.mem_virtual_usage_seq),
            'mem_physical': list(self.mem_physical_usage_seq),
            'io_read': list(self.io_read_seq),
            'io_write': list(self.io_write_seq),
        }

    def get_monitor_data_avg(self):
        [proc.join() for proc in self.processes]
        cpu = list(self.cpu_usage_seq)
        mem_virtual = list(self.mem_virtual_usage_seq)
        mem_physical = list(self.mem_physical_usage_seq)
        io_read = list(self.io_read_seq)
        io_write = list(self.io_write_seq)

        avg_cpu = sum(cpu) / (len(cpu) + 1e-9) / self.n_cpu
        avg_read_io = sum(io_read) / (len(io_read) + 1e-9)
        avg_write_io = sum(io_write) / (len(io_write) + 1e-9)
        avg_virtual_memory = sum(mem_virtual) / (len(mem_virtual) + 1e-9)
        avg_physical_memory = sum(mem_physical) / (len(mem_physical) + 1e-9)
        return avg_cpu, avg_read_io, avg_write_io, avg_virtual_memory, avg_physical_memory

    def monitor_mem_usage(self):
        count = 0
        while self.alive.value and count < self.ticks:
            if count < self.warmup:
                time.sleep(self.interval)
                count = count + 1
                continue
            mem_physical = self.process.memory_info()[0]/(1024.0 * 1024.0 * 1024.0)
            mem_virtual = self.process.memory_info()[1]/(1024.0 * 1024.0 * 1024.0)
            self.mem_physical_usage_seq.append(mem_physical)
            self.mem_virtual_usage_seq.append(mem_virtual)
            time.sleep(self.interval)
            count += 1

    def monitor_io_usage(self):
        """
        Monitor IO usage for the target process.
        Falls back to disk-level I/O monitoring when process-level monitoring fails
        (e.g., in containers where /proc/<pid>/io is not accessible).
        """
        count = 0
        use_disk_fallback = False
        disk_fallback_warned = False
        
        while self.alive.value and count < self.ticks:
            if count < self.warmup:
                time.sleep(self.interval)
                count = count + 1
                continue
            
            # Try process-level I/O monitoring first
            io_measured = False
            if not use_disk_fallback:
                try:
                    sp1 = self.process.io_counters()
                    time.sleep(self.interval)
                    sp2 = self.process.io_counters()
                    self.io_read_seq.append((sp2[2]-sp1[2])/(1024.0 * 1024.0))
                    self.io_write_seq.append((sp2[3]-sp1[3])/(1024.0 * 1024.0))
                    io_measured = True
                except (psutil.AccessDenied, PermissionError) as e:
                    # Switch to disk-level fallback
                    use_disk_fallback = True
                    if not disk_fallback_warned:
                        logger.warning(
                            f"Cannot access process IO counters for PID {self.process.pid}: {e}. "
                            f"Falling back to disk-level I/O monitoring. This may occur in containers "
                            f"due to PID namespace isolation or kernel restrictions."
                        )
                        disk_fallback_warned = True
                    # Get initial disk counter before sleeping
                    try:
                        disk1 = psutil.disk_io_counters()
                        time.sleep(self.interval)
                        disk2 = psutil.disk_io_counters()
                        if disk1 and disk2:
                            read_bytes = (disk2.read_bytes - disk1.read_bytes) / (1024.0 * 1024.0)
                            write_bytes = (disk2.write_bytes - disk1.write_bytes) / (1024.0 * 1024.0)
                            self.io_read_seq.append(read_bytes)
                            self.io_write_seq.append(write_bytes)
                            io_measured = True
                        else:
                            self.io_read_seq.append(0.0)
                            self.io_write_seq.append(0.0)
                    except Exception as disk_e:
                        logger.warning(f"Disk-level I/O monitoring failed: {disk_e}")
                        self.io_read_seq.append(0.0)
                        self.io_write_seq.append(0.0)
                except psutil.NoSuchProcess:
                    logger.warning(f"Process {self.process.pid} no longer exists")
                    self.io_read_seq.append(0.0)
                    self.io_write_seq.append(0.0)
                    break
            
            # Use disk-level I/O monitoring as fallback (for subsequent iterations)
            if use_disk_fallback and not io_measured:
                try:
                    # Get system-wide disk I/O counters (works in containers)
                    disk1 = psutil.disk_io_counters()
                    time.sleep(self.interval)
                    disk2 = psutil.disk_io_counters()
                    
                    if disk1 and disk2:
                        # Calculate MB/s for read and write bytes
                        read_bytes = (disk2.read_bytes - disk1.read_bytes) / (1024.0 * 1024.0)
                        write_bytes = (disk2.write_bytes - disk1.write_bytes) / (1024.0 * 1024.0)
                        self.io_read_seq.append(read_bytes)
                        self.io_write_seq.append(write_bytes)
                    else:
                        # Disk counters not available
                        self.io_read_seq.append(0.0)
                        self.io_write_seq.append(0.0)
                except Exception as e:
                    logger.warning(f"Disk-level I/O monitoring also failed: {e}")
                    self.io_read_seq.append(0.0)
                    self.io_write_seq.append(0.0)
            
            count += 1

    def monitor_cpu_usage(self):
        count = 0
        while self.alive.value and count < self.ticks:
            if count < self.warmup:
                time.sleep(self.interval)
                count = count + 1
                continue
            cpu = self.process.cpu_percent(interval=self.interval)
            self.cpu_usage_seq.append(cpu)
            count += 1

    def terminate(self):
        self.alive.value = False
