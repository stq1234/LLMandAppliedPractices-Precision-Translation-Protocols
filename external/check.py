"""Small post-hoc transfer check on six attributed historical ISO excerpts."""
from pathlib import Path
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run_experiment import ChatClient, ResponseCache, DEFAULT_MODEL, _run_condition, _run_pool_pair

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    sources=json.loads((HERE/"sources.json").read_text(encoding="utf-8"))
    cache=ResponseCache(HERE.parent/"results"/"external_cache.sqlite3")
    client=ChatClient(DEFAULT_MODEL,cache)
    def one(s):
        direct=_run_condition(s,{},"direct","none",client)
        paired=_run_pool_pair(s,{},client)
        return [direct,*paired["result_records"]]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            runs=[row for group in pool.map(one,sources) for row in group]
    finally:cache.close()
    write("runs.json",runs)
    assert len(runs)==18 and all(r["status"]=="success" for r in runs)
    packets={}
    for r in runs:
        for s in [r["translation_zh"],*r["stages"].get("candidate_translations",[])]:
            k=sha256((r["item_id"]+"|"+s).encode()).hexdigest()[:12]
            packets[k]={"id":k,"source":r["source_en"],"zh":s}
    write("blinded.json",sorted(packets.values(),key=lambda x:x["id"]))
    print("conditions",len(runs),"physical_calls",sum(r["physical_calls"] for r in runs),"unique_for_review",len(packets))

if __name__=="__main__":main()
