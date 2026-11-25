"""Counts only: never reads private/ or prints config/prompt/credential values."""
import json
from pathlib import Path
import sqlite3


def main():
    root=Path(__file__).resolve().parents[1];result={'project':str(root),'builds':{},'experiments':{}}
    for folder in (root/'graph_builds').iterdir():
        if not(folder/'build.sqlite').exists():continue
        db=sqlite3.connect((folder/'build.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        result['builds'][folder.name]={'saved_summaries':db.execute("SELECT count(*) FROM artifacts WHERE key LIKE 'summary:%'").fetchone()[0],
            'model_requests':db.execute('SELECT count(*) FROM requests').fetchone()[0]}
        manifest=folder/'build_manifest.json'
        if manifest.exists():result['builds'][folder.name]['status']=json.loads(manifest.read_text()).get('status')
        db.close()
    for folder in(root/'experiments').iterdir():
        if not(folder/'experiment.sqlite').exists():continue
        db=sqlite3.connect((folder/'experiment.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        result['experiments'][folder.name]={status:count for status,count in db.execute('SELECT status,count(*) FROM tasks GROUP BY status')};db.close()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
