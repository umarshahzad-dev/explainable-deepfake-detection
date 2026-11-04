from PIL import Image
import torch
from tests.support import module, workspace


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3,3)
        self.classifier = torch.nn.Linear(3,1)
    def get_classifier(self): return self.classifier
    def forward(self,x): return self.classifier(self.backbone(x.mean((2,3))))


def crop_fixture(test, splits=("train","val","calibration","test"), count=1):
    paths = module(test,"paths")
    ws = paths.Workspace(workspace(test))
    cfg = module(test,"config").load_config(root=ws.root, overrides={"training":{"head_epochs":1,"finetune_epochs":1,"workers":0,"epoch_samples":4,"batch_size":2,"accumulation":2,"amp":False,"augmentation":"none"}})
    contract = module(test,"preprocessing").preprocessing_hash(cfg)
    rows=[]
    for partition,split in enumerate(splits):
        for label in (0,1):
            for index in range(count):
                key=f"{split}-{label}-{index}"
                path=ws.crops/(key+".png")
                Image.new("RGB",(299,299),(20+label*150,20+partition*20,index%255)).save(path)
                rows.append({"sample_id":key,"video_id":key,"dataset":"ffpp","method":"original" if label else "Deepfakes",
                             "label":label,"split":split,"parents":[key],"known_person_ids":[],"identity_verified":False,
                             "sha256":"source-"+key,"crop_path":str(path),"crop_sha256":paths.digest_file(path),
                             "preprocessing_hash":contract,"master_size":299,"extraction_status":"completed","subject_verified":True,
                             "verification_source":"manual_registry","cluster":key})
    paths.write_jsonl(ws.manifest("fixture")/"crops.jsonl",rows,ws)
    return ws,cfg,rows
