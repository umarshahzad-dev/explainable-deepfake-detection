"""Human summaries and a shell-session banner; audit summaries persist in quiet mode."""
import os
from pathlib import Path
from deepfake_detection.common.paths import Workspace,fingerprint,write_json
from deepfake_detection.common.progress import _quiet,duration,is_test_mode


def extraction_detail(crops,target,auto,review):
    return f"crops {crops}/{target} · auto-verified {auto} · review-needed {review}"


def shell_identity():
    parent=os.getppid()
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes as w
        class Entry(ctypes.Structure):
            _fields_=[("dwSize",w.DWORD),("cntUsage",w.DWORD),("th32ProcessID",w.DWORD),("th32DefaultHeapID",ctypes.c_size_t),
                      ("th32ModuleID",w.DWORD),("cntThreads",w.DWORD),("th32ParentProcessID",w.DWORD),("pcPriClassBase",w.LONG),
                      ("dwFlags",w.DWORD),("szExeFile",w.WCHAR*260)]
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.CreateToolhelp32Snapshot.restype=w.HANDLE
        kernel.Process32FirstW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)]
        kernel.Process32NextW.argtypes=[w.HANDLE,ctypes.POINTER(Entry)]
        kernel.CloseHandle.argtypes=[w.HANDLE]
        snapshot=kernel.CreateToolhelp32Snapshot(2,0)
        processes={}
        if snapshot and snapshot != ctypes.c_void_p(-1).value:
            entry=Entry();entry.dwSize=ctypes.sizeof(entry)
            ok=kernel.Process32FirstW(snapshot,ctypes.byref(entry))
            while ok:
                processes[entry.th32ProcessID]=(entry.th32ParentProcessID,entry.szExeFile.lower())
                ok=kernel.Process32NextW(snapshot,ctypes.byref(entry))
            kernel.CloseHandle(snapshot)
        visited=set()
        while parent in processes and parent not in visited:
            visited.add(parent)
            ancestor,name=processes[parent]
            if name in {"powershell.exe","pwsh.exe","cmd.exe","bash.exe"}: break
            if not ancestor: break
            parent=ancestor
        kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
        handle=kernel.OpenProcess(0x1000,False,parent)
        creation=w.FILETIME();exit_time=w.FILETIME();cpu=w.FILETIME();user=w.FILETIME()
        kernel.GetProcessTimes.argtypes=[w.HANDLE,*([ctypes.POINTER(w.FILETIME)]*4)]
        token="unknown"
        if handle:
            if kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(cpu),ctypes.byref(user)):
                token=str((creation.dwHighDateTime<<32)|creation.dwLowDateTime)
            kernel.CloseHandle(handle)
        return f"{parent}:{token}"
    try:
        start=Path(f"/proc/{parent}/stat").read_text().rsplit(")",1)[1].split()[19]
    except (OSError,IndexError): start="unknown"
    return f"{parent}:{start}"


def first_banner(config,device,workspace=None,*,shell_token=None,enabled=None):
    enabled=(not is_test_mode() if enabled is None else enabled) and not _quiet.get()
    if not enabled: return
    ws=workspace or Workspace()
    marker=ws.runs/".sessions"/(fingerprint(shell_token or shell_identity())+".json")
    if marker.exists(): return
    print(f"experiment/ pipeline · config={config} · device={device}",flush=True)
    write_json(marker,{"config":str(config),"device":str(device)},ws)


def format_summary(command,result):
    overall=result.get("overall") or {}
    if command=="extract":
        pairs=[("Crops saved",result.get("saved_crops")),("Splits touched",sorted({r["split"] for r in result.get("counts",[])})),
               ("Auto-verified",result.get("auto_verified_crops")),("Review-needed",result.get("review_required_crops")),
               ("Disk used",f"{result.get('physical_png_bytes',result.get('stored_png_bytes',0))/2**20:.2f} MiB"),("Elapsed",duration(result.get("elapsed_seconds",0)))]
    elif command=="train":
        pairs=[("Best val AUC",result.get("best_validation_macro_auc")),("Epochs run",result.get("epochs")),
               ("Optimizer updates",result.get("optimizer_updates")),("Elapsed",duration(result.get("elapsed_seconds",0))),("Best checkpoint",result.get("best_checkpoint_path"))]
    elif command=="evaluate":
        pairs=[("Per-dataset AUC",{k:v.get("roc_auc") for k,v in result.get("per_dataset",{}).items()}),
               ("Fake recall at 1% FPR",overall.get("fake_recall_at_real_fpr_1pct")),("Fake recall at 5% FPR",overall.get("fake_recall_at_real_fpr_5pct")),
               ("Elapsed",duration(result.get("elapsed_seconds",0))),("Report path",result.get("report_path"))]
    elif command=="calibrate":
        pairs=[("Temperature",result.get("temperature")),("Threshold",result.get("threshold")),
               ("Calibration FPR",result.get("calibration_real_fpr")),("Calibration recall",result.get("calibration_fake_recall"))]
    elif command=="benchmark":
        pairs=[("Images/second",result.get("images_per_second")),("Data loading seconds",result.get("dataset_load_seconds")),
               ("Augmentation seconds",result.get("augmentation_seconds")),("Forward/backward seconds",result.get("forward_backward_seconds")),
               ("Peak allocated MiB",result.get("peak_allocated_mib")),("Sampled process MiB",result.get("vram_sampled_mib")),("Report path",result.get("report_path"))]
    elif command=='find-lr':
        pairs=[('Optimizer updates',result.get('optimizer_updates')),('Suggested backbone LR',result.get('recommended_lr')),('Elapsed',duration(result.get('elapsed_seconds',0))),('Report path',result.get('report_path'))]
    elif command=="identity-audit":
        pairs=[("Verified records",result.get("person_disjoint_verified_records")),("Unverified records",result.get("person_unverified_records")),
               ("Verified-person overlaps",result.get("verified_person_overlap")),("Report path",result.get("report_path"))]
    else:
        pairs=[(key,result.get(key)) for key in ("admitted","quarantined","records","elapsed_seconds","report_path") if key in result]
    return "Summary: "+command+"\nMetric | Value\n"+"\n".join(f"{key} | {value}" for key,value in pairs)


def print_summary(command,result,*,enabled=None):
    text=format_summary(command,result)
    enabled=(not is_test_mode() if enabled is None else enabled) and not _quiet.get()
    if enabled: print(text,flush=True)
    else:
        from deepfake_detection.common.run_log import audit_message
        audit_message("SUMMARY",text)
