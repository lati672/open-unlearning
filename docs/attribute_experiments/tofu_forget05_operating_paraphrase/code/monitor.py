"""Record measured GPU occupancy without interfering with any process."""
import time,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
while True:
 status=json.loads((R/'status.json').read_text())
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
 rows=[]
 for line in raw.strip().splitlines():
  i,name,u,mem,power=[x.strip() for x in line.split(',')]
  rows.append(dict(index=int(i),name=name,utilization_percent=float(u),memory_mib=float(mem),power_watts=float(power)))
 with (R/'logs/gpu_telemetry.jsonl').open('a') as f:f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),monotonic=time.monotonic(),status=status,gpus=rows))+'\n')
 if status['phase'] in ['all_measurements_complete','failed']:break
 time.sleep(2)
