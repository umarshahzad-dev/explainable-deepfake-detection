"""Generate explicitly unresolved review templates; never creates a release."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from .split import file_sha256


def generate(draft,components,component_report,output):
    draft=Path(draft);components=Path(components);output=Path(output);metadata=json.loads(Path(component_report).read_text(encoding='utf-8'))
    if metadata.get('component_csv_sha256')!=file_sha256(components):raise ValueError('Component mapping hash mismatch')
    mapping={}
    with components.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            group=row.get('group_id','');component=row.get('component_id','')
            if not group or not component or group in mapping:raise ValueError('Invalid component mapping')
            mapping[group]=component
    entries=defaultdict(lambda:{'datasets':set(),'partitions':set(),'records':0});missing=0
    with draft.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            part=row.get('proposed_partition','')
            if part not in {'train','val','calibration','test'}:continue
            component=mapping.get(row.get('group_id',''))
            if not component:missing+=1;continue
            item=entries[component];item['datasets'].add(row.get('dataset',''));item['partitions'].add(part);item['records']+=1
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('component-reviews.csv','cross-media-review.json','review-template-report.json')):
        raise FileExistsError('Review template exists; choose a new output directory to preserve decisions')
    fields=['component_id','decision','reviewer','evidence_path','evidence_sha256','datasets','proposed_partitions','records']
    with (output/'component-reviews.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for component,item in sorted(entries.items()):
            writer.writerow({'component_id':component,'decision':'unresolved','reviewer':'','evidence_path':'','evidence_sha256':'','datasets':'|'.join(sorted(item['datasets'])),'proposed_partitions':'|'.join(sorted(item['partitions'])),'records':item['records']})
    cross={'status':'incomplete','scope':'all_active_cross_partition_source_pairs','covered_cross_dataset_pairs':[],'manifest_sha256':file_sha256(draft),'component_csv_sha256':metadata['component_csv_sha256'],'unresolved_count':len(entries)+missing,'reviewer':'','reviewed_components':[],'evidence_files':[],'limitations':['Template is not evidence','Exact byte hashing and limited perceptual probes do not prove source independence']}
    (output/'cross-media-review.json').write_text(json.dumps(cross,indent=2),encoding='utf-8')
    summary={'draft_sha256':cross['manifest_sha256'],'active_proposed_components':len(entries),'candidate_records_without_component':missing,'status':'UNRESOLVED_TEMPLATE_NOT_EVIDENCE'}
    (output/'review-template-report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--draft',required=True,type=Path);p.add_argument('--components',required=True,type=Path);p.add_argument('--component-report',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(json.dumps(generate(a.draft,a.components,a.component_report,a.output),indent=2))

if __name__=='__main__':main()
