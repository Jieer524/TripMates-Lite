"""Capture the latency protocol and host metadata used by the benchmark."""
from pathlib import Path
import json, platform, os, subprocess, sys
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"output/research_final"
def cmd(s):
 try:return subprocess.check_output(s,shell=True,text=True,stderr=subprocess.DEVNULL).strip()
 except Exception:return None
def main():
 ram=cmd("powershell -NoProfile -Command \"[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)\"")
 logical=os.cpu_count(); physical=cmd("powershell -NoProfile -Command \"(Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors\"")
 payload={"hardware":{"cpu":platform.processor(),"logical_cpu_count":logical,"reported_logical_processors":physical,"ram_gb":ram,"storage_type":"not reliably observable from repository runtime; SSD/HDD not inferred"},"software":{"os":platform.platform(),"python":sys.version.split()[0]},"protocol":{"warmup_count":1,"timed_repetitions":30,"timing_scopes":["engine_only","end_to_end_request","shared_preparation"],"file_io_included":False,"weather_loading_included":False,"model_setup_included":False,"reference_route_generation":"included in shared preparation where applicable","worker_count":1,"research_configuration":"population 40, generations 60","quick_preview_configuration":"population 12, generations 8","fixed_seed":1001,"latency_unit":"seconds in raw files; milliseconds in thesis interpretation"},"source":"run_latency_benchmark.py"}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/"latency_protocol.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
 lines=["# Latency protocol","",f"- CPU: {payload['hardware']['cpu']}",f"- Logical CPUs: {logical}",f"- RAM (GB): {ram}",f"- Storage: {payload['hardware']['storage_type']}",f"- OS: {payload['software']['os']}",f"- Python: {payload['software']['python']}","- Warm-up: 1 untimed run","- Timed repetitions: 30","- Scopes: engine-only, end-to-end request, shared preparation","- File I/O and weather loading are outside the timed engine scope; problem preparation is reported separately.","- Research GA: population 40, 60 generations; quick preview: population 12, 8 generations.","- Worker count: 1; fixed seed: 1001.",""]
 (OUT/"latency_protocol.md").write_text("\n".join(lines),encoding="utf-8")
if __name__=="__main__":main()
