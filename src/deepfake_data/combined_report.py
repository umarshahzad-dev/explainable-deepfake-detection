"""Render the blocked allocation and bounded capacity estimate for review."""
import argparse
import collections
import json
import shutil
import statistics
from pathlib import Path


def generate(directory,output,quality_dir=None,stills_dir=None,proposal_name="proposed-report.json",audit_name="proposed-audit.json"):
    directory=Path(directory);r=json.loads((directory/proposal_name).read_text(encoding='utf-8'));a=json.loads((directory/audit_name).read_text(encoding='utf-8'));source=json.loads((directory/'source-link-report.json').read_text(encoding='utf-8'));links=json.loads((directory/'relationship-audit.json').read_text(encoding='utf-8'))
    totals=collections.Counter();reasons=collections.Counter();methods=collections.Counter();media=collections.Counter()
    for item in r['coverage']:
        p,d,l=item['proposed_partition'],item['dataset'],item['label'];n=item['count'];totals[(p,d,l)]+=n;reasons[(d,item['reason'])]+=n
        if p=='train':methods[(d,item['method'],l)]+=n
        if p!='quarantine':media[(d,item['media_type'])]+=n
    samples=[];samples_by_dataset=collections.defaultdict(list)
    for folder in (quality_dir,stills_dir):
        if folder and Path(folder).is_dir():
            for metadata in Path(folder).glob('*/complete.json'):
                data=json.loads(metadata.read_text(encoding='utf-8'));dataset=data.get('config',{}).get('provenance',{}).get('dataset','unknown')
                for frame in data.get('frames',[]):
                    path=metadata.parent/frame['file']
                    if path.is_file():size=path.stat().st_size;samples.append(size);samples_by_dataset[dataset].append(size)
    free=shutil.disk_usage(directory).free;videos=sum(n for (d,t),n in media.items() if t=='video');stills=sum(n for (d,t),n in media.items() if t in {'image','gif'});maximum=videos*20+stills;upper=maximum*(299*299*3*1.5+4096)
    lines=['# Combined allocation — BLOCKED DRAFT','','Every actual `project_partition` is `quarantine`. Proposed rows are not approved for extraction or training.',f"Input SHA-256: `{r['source_manifest_sha256']}`",f"Draft SHA-256: `{r['draft_manifest_sha256']}`",'','## Proposed coverage','','| Partition | Dataset | Fake | Real | Unknown |','|---|---|---:|---:|---:|']
    for p in ['train','val','calibration','test','quarantine']:
        for d in ['ffpp','celebdf','df40']:
            if sum(totals[(p,d,l)] for l in ('0','1','unknown')):lines.append(f"| {p} | {d} | {totals[(p,d,'0')]} | {totals[(p,d,'1')]} | {totals[(p,d,'unknown')]} |")
    lines+=['','## Proposed groups','','| Partition | Dataset | Candidate groups |','|---|---|---:|']
    for x in r['proposed_group_counts']:lines.append(f"| {x['partition']} | {x['dataset']} | {x['groups']} |")
    lines+=['','## Proposed training methods','','| Dataset | Method | Label | Media records |','|---|---|---|---:|']
    for (d,m,l),n in sorted(methods.items()):lines.append(f'| {d} | {m} | {l} | {n} |')
    lines+=['','## Proposal reasons','','| Dataset | Reason | Records |','|---|---|---:|']
    for (d,reason),n in sorted(reasons.items()):lines.append(f'| {d} | {reason} | {n} |')
    lines+=['','## Evidence and blockers','',f"- Selected DF40 metadata: {links['df40_test_val_comparison']['same_complete_records']} of {links['df40_test_val_comparison']['class_blocks']} class blocks have identical test and validation records. This demonstrates duplication but not the authors’ intent.",f"- DeepFakeDetection: {source['actors']} actors, {source['actor_components']} actor components, {source['issues'].get('source_scene_missing',0)} missing same-scene source originals. Components 05 and 10 (584 videos) are reserved as unverified evaluation candidates; cross-dataset independence is unknown.",f"- Celeb-DF: {links['celebdf_filename_evidence']['non_official_withheld_by_either_candidate_id']} non-official videos are withheld under the conservative two-ID rule; additional records can be withheld by suspected source components. The official list is reserved as test candidates, while all actual project assignments remain quarantined.",f"- Proposed records with no hashes embedded in the draft: {a['missing_hashes']}. Proposed source-group overlaps: {len(a['group_overlap'])}; embedded hash overlaps: {len(a['hash_overlap'])}. The linked sidecar hash audit is complete: {a.get('external_hash_evidence',{}).get('complete',False)}. Empty results are not proof when hashes or relationships are missing.",'- The proposed validation, calibration and test partitions are not verified independent from all proposed training material. No DF40 modern method is assigned to evaluation: 1,882 MidJourney real/fake records are reserved in quarantine pending evidence.', '- DF40 official testing-release files are only candidates for experimental training; the proposed experiment is not an official DF40 benchmark.', '- Proposed DF40 media are capped at 10 per original sample group and 5,000 per method and label; additional candidates are quarantined. Group and source provenance remain subject to review.', '- The sampling-weights tool proposes equal dataset, class, method and group weighting for the future training manifest. No training sampler has been run.', '', '## Capacity estimate','',f'- Disk free: {free/1024**3:.2f} GiB; proposed video files: {videos}; proposed DF40 still/GIF files: {stills}.',f'- Up to {maximum:,} images, assuming at most 20 per video and one per GIF; the earlier structural audit found one image frame in each of the 247 GIFs.',f'- Conservative 1.5× raw RGB allowance plus per-file overhead: {upper/1024**3:.2f} GiB, excluding logs and safety reserve.']
    if samples:
        means={d:statistics.mean(v) for d,v in samples_by_dataset.items()}
        if all(d in means for d,t in media):
            estimate=sum(n*(20 if kind=='video' else 1)*means[d] for (d,kind),n in media.items())
            lines.append(f'- Dataset-weighted estimate from {len(samples)} sample PNGs: {estimate/1024**3:.2f} GiB. This small sample cannot guarantee available space.')
        else:lines.append(f'- Mean of {len(samples)} sample PNGs: {statistics.mean(samples):,.0f} bytes; samples do not cover every source dataset.')
    component_report=directory/'relationships/source-components-report.json'
    if component_report.is_file():
        component=json.loads(component_report.read_text(encoding='utf-8'))
        lines.append(f"- Source graph: {component['suspected_source_edges']:,} suspected source edges, {component['components_with_multiple_datasets']} multi-dataset components and {component['suspected_identity_memberships']:,} separate suspected identity memberships. These are not verified identity links.")
    release_report=directory/'release-preflight-v2.json'
    if release_report.is_file():
        release=json.loads(release_report.read_text(encoding='utf-8'));lines.append(f"- Release preflight ready: {release['ready']}; blockers: {'; '.join(release['blockers'])}.")
    duplicate_sample=directory/'duplicate-sample-summary.json'
    if duplicate_sample.is_file():
        summary=json.loads(duplicate_sample.read_text(encoding='utf-8'))
        lines.append(f"- Bounded byte-hash sample: {summary['unique_source_paths']} unique files, including a 60-record stratified sample; {len(summary['cross_partition_exact_hash_groups'])} cross-partition exact matches. This does not establish complete coverage.")
    timing=directory/'storage-estimate-v2.json'
    if timing.is_file():
        result=json.loads(timing.read_text(encoding='utf-8'))
        lines.append(f"- Four sampled videos with four candidate frames each took {result['video_sample_seconds']} seconds including startup. This does not establish full-run throughput.")
    lines+=['- Full-workload throughput and peak GPU memory have not been measured. No bulk processing has begun.']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text('\n'.join(lines)+'\n',encoding='utf-8');return {'videos':videos,'stills':stills,'max_images':maximum,'free_bytes':free,'conservative_bytes':int(upper)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--quality-dir',type=Path);p.add_argument('--stills-dir',type=Path);p.add_argument('--proposal-name',default='proposed-report.json');p.add_argument('--audit-name',default='proposed-audit.json');a=p.parse_args();print(json.dumps(generate(a.directory,a.output,a.quality_dir,a.stills_dir,a.proposal_name,a.audit_name),indent=2))

if __name__=='__main__':main()
