import time
import numpy as np
from direct.showbase.ShowBase import ShowBase
from gigades import GigaDES

TOTAL_VALUES = 2**34 
BATCH_SIZE   = 4_000_000 

class GigaBenchmarker(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        self.gdes = GigaDES(self)
        self.test_block = b'kgs!@#$%' 
        
    def format_time(self, seconds):
        """Format seconds into a readable string"""
        if seconds < 60:
            return f"{seconds:.3f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {mins}m {secs}s"
    
    def run_benchmark(self):
        total_done = 0
        key_start = 0x0000000000000000
        start_time = time.time()
        cycle_start = start_time

        print(f"Starting benchmark: {TOTAL_VALUES:,} total operations")
        print(f"Batch size: {BATCH_SIZE:,}")
        print("-" * 80)

        try:
            while total_done < TOTAL_VALUES:
                batch_len = min(BATCH_SIZE, TOTAL_VALUES - total_done)
                current_key = key_start + total_done
                
                # Time this cycle
                cycle_start = time.time()
                results = self.gdes.process_block(self.test_block, batch_len, key_start=current_key)
                cycle_time = time.time() - cycle_start
                
                total_done += batch_len
                elapsed = time.time() - start_time
                throughput = total_done / elapsed
                
                # Calculate progress and ETA
                progress_pct = (total_done / TOTAL_VALUES) * 100
                
                if total_done > 0 and throughput > 0:
                    remaining = TOTAL_VALUES - total_done
                    eta_seconds = remaining / throughput
                    eta_str = self.format_time(eta_seconds)
                else:
                    eta_str = "calculating..."
                
                # Print update every batch
                print(f"Cycle: {self.format_time(cycle_time):>8} | "
                      f"Total: {self.format_time(elapsed):>10} | "
                      f"Progress: {progress_pct:6.2f}% | "
                      f"Speed: {throughput:>12,.0f} ops/sec | "
                      f"ETA: {eta_str:>10}")

                base.taskMgr.step()

        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print("INTERRUPTED")
            elapsed = time.time() - start_time
            print(f"Completed: {total_done:,} / {TOTAL_VALUES:,} operations ({total_done/TOTAL_VALUES*100:.2f}%)")
            print(f"Total time: {self.format_time(elapsed)}")
            print(f"Average throughput: {total_done/elapsed:,.0f} ops/sec")
            print("=" * 80)
            return

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETE")
        print(f"Total operations: {total_done:,}")
        print(f"Total time: {self.format_time(total_time)}")
        print(f"Average throughput: {total_done/total_time:,.0f} ops/sec")
        print("=" * 80)

if __name__ == "__main__":
    app = GigaBenchmarker()
    app.run_benchmark()
