"""Audit source-name evidence and source JSON partition duplication."""
from __future__ import annotations
import argparse
import json
import re
from collections import Counter,defaultdict
from pathlib import Path
from .combined_draft import celeb_source_key,celeb_source_keys
from .inventory import metadata_files
from .source_links import dfd_components


def audit(root):
    root=Path(root);celeb=root/'celebdf/Celeb-DF-v2';df=root/'df40';lines=(celeb/'List_of_testing_videos.txt').read_text(encoding='utf-8-sig').splitlines();official={line.split(maxsplit=1)[1] for line in lines if len(line.split(maxsplit=1))==2}
    groups=defaultdict(Counter);shape=Counter();parents={};official_candidate_keys=set()
    for relative in official:
        official_candidate_keys.update(celeb_source_keys(Path(relative).stem))
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    for subset in ('Celeb-real','Celeb-synthesis'):
        for path in (celeb/subset).glob('*.mp4'):
            key=celeb_source_key(path.stem)
            is_official=f'{subset}/{path.name}' in official
            if key:groups[key]['official' if is_official else 'other']+=1
            if not is_official and celeb_source_keys(path.stem) & official_candidate_keys:
                shape['non_official_withheld_by_either_candidate_id']+=1
            if subset=='Celeb-synthesis':
                match=re.fullmatch(r'(id\d+)_(id\d+)_(\d+)',path.stem)
                if not match:shape['unparsed_fake']+=1;continue
                a,b,seq=match.groups();left,right=find(a),find(b);parents[max(left,right)]=min(left,right)
                shape['fake_names']+=1
                if (celeb/'Celeb-real'/f'{a}_{seq}.mp4').is_file():shape['first_id_and_sequence_match_real']+=1
                if (celeb/'Celeb-real'/f'{b}_{seq}.mp4').is_file():shape['second_id_and_sequence_match_real']+=1
    mixed={key:c for key,c in groups.items() if c['official'] and c['other']}
    shape.update(source_keys=len(groups),mixed_official_and_other_groups=len(mixed),non_official_videos_in_mixed_groups=sum(c['other'] for c in mixed.values()),official_videos_in_mixed_groups=sum(c['official'] for c in mixed.values()))
    components=Counter(find(x) for x in list(parents))
    methods={p.name for p in (df/'extracted').iterdir() if p.is_dir()};dup=Counter();different=[]
    for file in metadata_files(df/'metadata/dataset_json',methods):
        data=json.loads(file.read_text(encoding='utf-8-sig'))
        for collection,classes in data.items():
            for cls,partitions in classes.items():
                test=partitions.get('test',{});val=partitions.get('val',{});dup['class_blocks']+=1;dup['same_sample_keys']+=set(test)==set(val);dup['same_complete_records']+=test==val;dup['shared_sample_keys']+=len(set(test)&set(val))
                if test!=val and len(different)<10:different.append([file.name,collection,cls])
    _,dfd=dfd_components(root/'ffpp')
    return {'celebdf_filename_evidence':dict(shape),'celebdf_candidate_identity_component_sizes':sorted(components.values(),reverse=True),'df40_test_val_comparison':dict(dup),'df40_different_blocks_examples':different,'dfd':dfd,'limitations':['Celeb-DF filename matching is conservatively treated as suspected linkage, not proof of target/source semantics', 'Connected actor/identity components do not prove identities are absent from other datasets', 'The reason for repeated DF40 test/val records cannot be established solely from these files', 'No person-identity independence or complete duplicate check is claimed']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',default=r'D:\University\datasets\deepfake',type=Path);p.add_argument('--report',required=True,type=Path);a=p.parse_args();result=audit(a.dataset_root);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
