"""Reallocate an already audited draft using conservative source components.

Only records eligible in the input draft can be promoted. Unknown and excluded
records stay quarantined. Exact-byte conflicts quarantine whole components.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .extract import validate_output
from .split import file_sha256

ACTIVE = {'train', 'val', 'calibration', 'test'}


def build(draft, proposal, components, component_report, hashes, hash_report, output, report, seed=42):
    draft, output = Path(draft), Path(output)
    validate_output(output.parent); validate_output(Path(report).parent)
    if draft.resolve() == output.resolve() or output.exists():
        raise ValueError('Choose a new allocation output')
    original = json.loads(Path(proposal).read_text(encoding='utf-8'))
    evidence = json.loads(Path(component_report).read_text(encoding='utf-8'))
    audit = json.loads(Path(hash_report).read_text(encoding='utf-8'))
    if (original.get('draft_manifest_sha256') != file_sha256(draft)
        or evidence.get('manifest_sha256') != original.get('source_manifest_sha256')
        or evidence.get('component_csv_sha256') != file_sha256(components)
        or not audit.get('complete') or not audit.get('proposal_mode')
        or audit.get('manifest_sha256') != file_sha256(draft)
        or audit.get('hash_csv_sha256') != file_sha256(hashes) or audit.get('errors')):
        raise ValueError('Input proposal, source components or complete hash audit do not match')
    mapping = {}
    with Path(components).open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not row['group_id'] or row['group_id'] in mapping: raise ValueError('Invalid component mapping')
            mapping[row['group_id']] = row['component_id']
    anchors = defaultdict(set); has_nonofficial_celeb = set(); paths = {}; n = 0
    with draft.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('proposed_partition') not in ACTIVE: continue
            component = mapping.get(row['group_id'])
            if not component or component != row.get('proposed_source_group'): raise ValueError('Missing or inconsistent component')
            path = str(Path(row['absolute_path']).resolve()).casefold()
            if path in paths: raise ValueError('Duplicate active source path')
            paths[path] = component; n += 1
            if row['dataset'] == 'ffpp': anchors[component].add(row['proposed_partition'])
            if row['dataset'] == 'celebdf':
                if row.get('original_partition') == 'official_test': anchors[component].add('test')
                else: has_nonofficial_celeb.add(component)
    assignment = {}
    for component in set(paths.values()):
        fixed = anchors[component]
        if len(fixed) > 1 or ('test' in fixed and component in has_nonofficial_celeb):
            assignment[component] = 'quarantine'; continue
        if fixed: assignment[component] = next(iter(fixed)); continue
        value = int.from_bytes(hashlib.sha256(f'{seed}:{component}'.encode()).digest()[:8], 'big') / 2**64
        # The official Celeb-DF test list is the only Celeb-DF test source.
        if component in has_nonofficial_celeb:
            assignment[component] = 'train' if value < .8 else 'val' if value < .9 else 'calibration'
        else:
            assignment[component] = 'train' if value < .7 else 'val' if value < .8 else 'calibration' if value < .9 else 'test'
    digest_groups = defaultdict(set); seen = set()
    with Path(hashes).open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            path = str(Path(row['absolute_path']).resolve()).casefold()
            if path not in paths or path in seen or row.get('status') != 'ok': raise ValueError('Hash audit does not cover exactly the original active paths')
            seen.add(path); digest_groups[row['sha256']].add(paths[path])
    if len(seen) != n: raise ValueError('Hash audit coverage incomplete')
    conflicting = set()
    for groups in digest_groups.values():
        if len({assignment[g] for g in groups} - {'quarantine'}) > 1: conflicting.update(groups)
    for group in conflicting: assignment[group] = 'quarantine'
    counts = Counter(); output.parent.mkdir(parents=True, exist_ok=True)
    with draft.open(newline='', encoding='utf-8') as src, output.open('x', newline='', encoding='utf-8') as dst:
        reader = csv.DictReader(src); fields = list(reader.fieldnames or [])
        if 'proposal_reason' not in fields: fields.append('proposal_reason')
        writer = csv.DictWriter(dst, fieldnames=fields); writer.writeheader()
        for row in reader:
            if row.get('proposed_partition') in ACTIVE:
                component = mapping[row['group_id']]; part = assignment[component]
                row['proposed_partition'] = part
                row['proposal_reason'] = ('exact_hash_or_source_anchor_conflict' if part == 'quarantine' else 'best_effort_source_component_allocation')
            counts[(row.get('proposed_partition',''), row.get('dataset',''), row.get('method',''), row.get('label',''))] += 1
            writer.writerow(row)
    result = {'status':'DRAFT_BLOCKED_NOT_FOR_EXTRACTION_OR_TRAINING',
              'config':dict(original['config'], version='combined-best-effort-v1', seed=seed, input_draft_sha256=file_sha256(draft)),
              'source_manifest_sha256':original['source_manifest_sha256'], 'draft_manifest_sha256':file_sha256(output),
              'previous_complete_hash_audit_sha256':file_sha256(hash_report), 'exact_conflict_components_quarantined':len(conflicting),
              'coverage':[{'partition':p,'dataset':d,'method':m,'label':l,'count':n} for (p,d,m,l),n in sorted(counts.items())],
              'limitations':['Known and suspected source links use existing conservative components; unknown identity is not verified', 'DF40 testing release is repurposed experimentally', 'Exact hashes do not detect re-encoded or cropped source content']}
    Path(report).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def supplement(draft, proposal, components, output, report, per_method_partition=50, per_group=5):
    """Co-locate a bounded set of FF-derived DF40 frames with held-out FF sources.

    These records were quarantined specifically because their FF source was
    held out. The component proves the co-location; the release still requires
    a complete new exact-byte audit, so this cannot itself activate records.
    """
    draft, output = Path(draft), Path(output)
    validate_output(output.parent); validate_output(Path(report).parent)
    if output.exists() or draft.resolve() == output.resolve(): raise ValueError('Choose a new allocation output')
    prior=json.loads(Path(proposal).read_text(encoding='utf-8'))
    if prior.get('draft_manifest_sha256') != file_sha256(draft): raise ValueError('Proposal mismatch')
    mapping={}
    with Path(components).open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['group_id'] in mapping: raise ValueError('Duplicate component')
            mapping[r['group_id']]=r['component_id']
    anchors=defaultdict(set)
    with draft.open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('dataset')=='ffpp' and r.get('proposed_partition') in ACTIVE:
                anchors[mapping[r['group_id']]].add(r['proposed_partition'])
    selected=Counter(); groups=Counter(); coverage=Counter()
    with draft.open(newline='',encoding='utf-8') as src, output.open('x',newline='',encoding='utf-8') as dst:
        reader=csv.DictReader(src);fields=list(reader.fieldnames or [])
        if 'proposal_reason' not in fields:fields.append('proposal_reason')
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for r in reader:
            component=mapping.get(r.get('group_id','')); parts=anchors.get(component,set())
            if (r.get('dataset')=='df40' and r.get('proposal_reason')=='suspected_ffpp_heldout_overlap'
                and r.get('metadata_status') in {'matched','partition_conflict'} and r.get('label') in {'0','1'}
                and r.get('proposed_source_group')==component and len(parts)==1):
                part=next(iter(parts));key=(r.get('method',''),r['label'],part);group_key=key+(r['group_id'],)
                if part in {'val','calibration','test'} and selected[key]<per_method_partition and groups[group_key]<per_group:
                    r['proposed_partition']=part;r['proposal_reason']='ff_source_component_colocated_with_heldout_ffpp'
                    selected[key]+=1;groups[group_key]+=1
            coverage[(r.get('proposed_partition',''),r.get('dataset',''),r.get('method',''),r.get('label',''))]+=1
            writer.writerow(r)
    prior['draft_manifest_sha256']=file_sha256(output)
    prior['config'].update({'supplement_source_sha256':file_sha256(draft),'ff_derived_per_method_partition':per_method_partition,'ff_derived_per_group':per_group})
    prior['coverage']=[{'partition':p,'dataset':d,'method':m,'label':l,'count':n} for (p,d,m,l),n in sorted(coverage.items())]
    prior['supplemented_records']=sum(selected.values())
    Path(report).write_text(json.dumps(prior,indent=2),encoding='utf-8')
    return prior


def quarantine_conflicts(draft, proposal, hashes, hash_report, output, report):
    """Quarantine every whole source component touched by an exact hash conflict."""
    draft, output=Path(draft),Path(output)
    validate_output(output.parent);validate_output(Path(report).parent)
    if output.exists() or draft.resolve()==output.resolve():raise ValueError('Choose a new allocation output')
    prior=json.loads(Path(proposal).read_text(encoding='utf-8'))
    audit=json.loads(Path(hash_report).read_text(encoding='utf-8'))
    if (prior.get('draft_manifest_sha256')!=file_sha256(draft) or not audit.get('complete')
        or not audit.get('proposal_mode') or audit.get('manifest_sha256')!=file_sha256(draft)
        or audit.get('hash_csv_sha256')!=file_sha256(hashes) or audit.get('errors')):
        raise ValueError('Conflicting-hash evidence does not match draft')
    digest_groups=defaultdict(set);digest_parts=defaultdict(set)
    with Path(hashes).open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r.get('status')!='ok' or not r.get('group_id') or r.get('partition') not in ACTIVE:raise ValueError('Invalid hash record')
            digest_groups[r['sha256']].add(r['group_id']);digest_parts[r['sha256']].add(r['partition'])
    conflicts={digest for digest,parts in digest_parts.items() if len(parts)>1}
    groups=set().union(*(digest_groups[digest] for digest in conflicts)) if conflicts else set()
    coverage=Counter()
    with draft.open(newline='',encoding='utf-8') as src,output.open('x',newline='',encoding='utf-8') as dst:
        reader=csv.DictReader(src);fields=list(reader.fieldnames or [])
        if 'proposal_reason' not in fields:fields.append('proposal_reason')
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for r in reader:
            if r.get('proposed_partition') in ACTIVE and r.get('proposed_source_group') in groups:
                r['proposed_partition']='quarantine';r['proposal_reason']='cross_partition_exact_hash_component_conflict'
            coverage[(r.get('proposed_partition',''),r.get('dataset',''),r.get('method',''),r.get('label',''))]+=1
            writer.writerow(r)
    prior['config']['conflict_input_sha256']=file_sha256(draft)
    prior['config']['conflict_hash_report_sha256']=file_sha256(hash_report)
    prior['draft_manifest_sha256']=file_sha256(output)
    prior['exact_conflict_hashes_quarantined']=len(conflicts);prior['additional_conflict_components_quarantined']=len(groups)
    prior['coverage']=[{'partition':p,'dataset':d,'method':m,'label':l,'count':n} for (p,d,m,l),n in sorted(coverage.items())]
    Path(report).write_text(json.dumps(prior,indent=2),encoding='utf-8');return prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('draft','proposal','components','output','report'): p.add_argument('--'+name, required=True, type=Path)
    for name in ('component-report','hashes','hash-report'): p.add_argument('--'+name,type=Path)
    p.add_argument('--supplement',action='store_true');p.add_argument('--quarantine-conflicts',action='store_true');p.add_argument('--seed',type=int,default=42); a = p.parse_args()
    if a.quarantine_conflicts:
        if not a.hashes or not a.hash_report:p.error('Conflict quarantine requires --hashes and --hash-report')
        r=quarantine_conflicts(a.draft,a.proposal,a.hashes,a.hash_report,a.output,a.report)
        print(json.dumps({'draft_sha256':r['draft_manifest_sha256'],'conflict_hashes':r['exact_conflict_hashes_quarantined'],'components':r['additional_conflict_components_quarantined']},indent=2));return
    if a.supplement:
        r=supplement(a.draft,a.proposal,a.components,a.output,a.report)
        print(json.dumps({'draft_sha256':r['draft_manifest_sha256'],'supplemented_records':r['supplemented_records']},indent=2));return
    if not all((a.component_report,a.hashes,a.hash_report)):p.error('Initial allocation requires --component-report, --hashes and --hash-report')
    r = build(a.draft,a.proposal,a.components,a.component_report,a.hashes,a.hash_report,a.output,a.report,a.seed)
    print(json.dumps({'draft_sha256':r['draft_manifest_sha256'], 'exact_conflict_components_quarantined':r['exact_conflict_components_quarantined']},indent=2))


if __name__ == '__main__': main()
