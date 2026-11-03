"""Build conservative source components and separate suspected identity evidence."""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
from .inventory import ffpp_source_groups
from .split import file_sha256

CELEB_FAKE=re.compile(r"(id\d+)_(id\d+)_(\d+)")
CELEB_REAL=re.compile(r"(id\d+)_(\d+)")
FF_SAMPLE=re.compile(r"(\d{3})(?:_(\d{3}))?")


def build(manifest,ffpp_root,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);ff_groups=ffpp_source_groups(Path(ffpp_root));nodes={};counts=Counter()
    with Path(manifest).open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            group=row.get('group_id','')
            if not group:continue
            counts[group]+=1
            nodes.setdefault(group,{k:row.get(k,'') for k in ('dataset','method','source_domain','sample_id','group_status','relationship_status')})
    parent={g:g for g in nodes}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    edges={};identity=set()
    def edge(a,b,relation):
        if a==b or a not in nodes or b not in nodes:return
        pair=(min(a,b),max(a,b),relation);edges[pair]='suspected';join(a,b)
    for group,row in nodes.items():
        dataset=row['dataset'];sample=row['sample_id'];domain=row['source_domain'].casefold();fake=CELEB_FAKE.fullmatch(sample);real=CELEB_REAL.fullmatch(sample)
        if dataset=='celebdf':
            if fake:
                for actor in (fake[1],fake[2]):
                    edge(group,'celebdf:'+actor+'_'+fake[3],'filename_candidate_source')
                    identity.add((actor,group,'filename_candidate_identity'))
            elif real:identity.add((real[1],group,'filename_candidate_identity'))
        elif dataset=='df40' and domain=='cdf':
            if fake:
                edge(group,'celebdf:'+sample,'same_named_manipulated_sample')
                for actor in (fake[1],fake[2]):
                    edge(group,'celebdf:'+actor+'_'+fake[3],'cdf_candidate_source')
                    identity.add((actor,group,'cdf_candidate_identity'))
            elif real:
                edge(group,'celebdf:'+sample,'cdf_candidate_source');identity.add((real[1],group,'cdf_candidate_identity'))
        elif dataset=='df40' and domain=='ff':
            match=FF_SAMPLE.fullmatch(sample)
            if match:
                ids=[x for x in match.groups() if x]
                for source in ids:
                    if source in ff_groups:edge(group,'ffpp:youtube:'+ff_groups[source],'ff_source_id_candidate')
    component_members=Counter(find(g) for g in nodes);suspected={find(a) for a,b,reason in edges};component_datasets=defaultdict(set)
    for group,row in nodes.items():component_datasets[find(group)].add(row['dataset'])
    with (output/'source-components.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['group_id','component_id','evidence_status','component_group_count','dataset','method','media_records'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for group,row in sorted(nodes.items()):
            component=find(group);w.writerow({'group_id':group,'component_id':component,'evidence_status':'contains_suspected_links' if component in suspected else 'unlinked_or_within_dataset_only','component_group_count':component_members[component],'dataset':row['dataset'],'method':row['method'],'media_records':counts[group]})
    with (output/'source-edges.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['left_group','right_group','relationship','status','evidence']);
        for (a,b,kind),status in sorted(edges.items()):w.writerow([a,b,kind,status,'Exact source-domain/name match; no identity or content proof'])
    with (output/'identity-candidates.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['identity_name','group_id','evidence']);w.writerows(sorted(identity))
    summary={'manifest_sha256':file_sha256(manifest),'component_csv_sha256':file_sha256(output/'source-components.csv'),'edge_csv_sha256':file_sha256(output/'source-edges.csv'),'identity_csv_sha256':file_sha256(output/'identity-candidates.csv'),'groups':len(nodes),'conservative_components':len(component_members),'suspected_source_edges':len(edges),'components_with_multiple_datasets':sum(len(v)>1 for v in component_datasets.values()),'suspected_identity_memberships':len(identity),'largest_component_group_count':max(component_members.values(),default=0),'limitations':['Edges are conservative name/domain candidates, not verified identities or proof that source frames match','Identity candidates are separate from source components and do not establish identity safety','Groups without an explicit candidate edge are not proven independent']}
    (output/'source-components-report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--ffpp-root',default=r'D:\University\datasets\deepfake\ffpp',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(json.dumps(build(a.manifest,a.ffpp_root,a.output),indent=2))

if __name__=='__main__':main()
