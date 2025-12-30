import time
import numpy as np
from direct.showbase.ShowBase import ShowBase
from gigades import GigaDES

TOTAL_VALUES = 2**33 
BATCH_SIZE   = 4_000_000 

class GigaBenchmarker(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        self.gdes = GigaDES(self)
        self.test_block = b'beefface' 
        
    def run_benchmark(self):
        total_done = 0
        key_start = 0x0000000000000001
        start_time = time.time()

        try:
            while total_done < TOTAL_VALUES:
                batch_len = min(BATCH_SIZE, TOTAL_VALUES - total_done)
                current_key = key_start + total_done
                
                results, _ = self.gdes.process_block(self.test_block, batch_len, key_start=current_key)
                
                total_done += batch_len
                elapsed = time.time() - start_time
                throughput = total_done / elapsed
                
                if (total_done // BATCH_SIZE) % 10 == 0:
                    print(f"Progress: {total_done/TOTAL_VALUES*100:6.2f}% | Speed: {throughput:,.0f} ops/sec")

                self.graphics_engine.render_frame()

        except KeyboardInterrupt:
            print("\nInterrupted.")

if __name__ == "__main__":
    app = GigaBenchmarker()
    app.run_benchmark()
