import io
import csv
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from tests.support import module,workspace


class IdentityAuditTests(unittest.TestCase):
    def rows(self,ws,verified=False):
        rows=[]
        for i in range(2):
            path=ws.root/f"raw{i}.mp4";path.write_bytes(bytes([i]))
            rows.append({"sample_id":str(i),"dataset":"ffpp","method":"original","label":1,"split":"train",
                         "parents":[f"p{i}"],"known_person_ids":[f"person{i}"] if verified or i==0 else [],
                         "identity_verified":verified or i==0,"raw_path":str(path)})
        return rows

    def test_mixed_audit_emits_csv_and_json(self):
        ws=module(self,"paths").Workspace(workspace(self));rows=self.rows(ws)
        module(self,"paths").write_jsonl(ws.manifest("m")/"videos.jsonl",rows,ws)
        result=module(self,"identity").identity_audit(["m"],ws)
        self.assertEqual(result["person_unverified_records"],1)
        with (ws.report("identity-audit")/"families.csv").open(newline="",encoding="utf-8") as file:
            records=list(csv.DictReader(file))
        self.assertEqual(set(records[0]),{"family_token","split","n_records","verified_person_ids","unverified_reason","recommended_action"})
        self.assertTrue((ws.report("identity-audit")/"summary.json").exists())

    def test_audit_cli_exit_codes_zero_and_two(self):
        from tools.laptop_workflow import main
        ws=module(self,"paths").Workspace(workspace(self))
        for verified,expected in ((False,2),(True,0)):
            module(self,"paths").write_jsonl(ws.manifest("m")/"videos.jsonl",self.rows(ws,verified),ws)
            with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(main(["identity-audit","--manifests","m"],workspace=ws),expected)

    def prepare_case(self,require=False):
        paths=module(self,"paths");ws=paths.Workspace(workspace(self));rows=self.rows(ws)
        for row in rows: row.update(identity_verified=False,known_person_ids=[])
        mapping=ws.root/"map.json";paths.write_json(mapping,{"person_id_map":{"p0":"external0"}},ws)
        cfg=module(self,"config").load_config(root=ws.root)
        with patch("deepfake_detection.data.manifests._ffpp",return_value=(rows,{"p0":"train","p1":"val"})):
            result=module(self,"manifests").prepare("ffpp",cfg,ws,identity_map=mapping,require_verified_identities=require)
        return paths,ws,result

    def test_prepare_injects_map_and_records_hash(self):
        paths,ws,result=self.prepare_case()
        rows=paths.read_jsonl(ws.manifest("ffpp")/"videos.jsonl")
        self.assertEqual(rows[0]["known_person_ids"],["external0"])
        self.assertTrue(rows[0]["identity_verified"])
        self.assertFalse(rows[1]["identity_verified"])
        self.assertEqual(result["identity_map_sha256"],paths.digest_file(ws.root/"map.json"))

    def test_require_verified_quarantines_unknown(self):
        paths,ws,result=self.prepare_case(True)
        rejected=paths.read_jsonl(ws.manifest("ffpp")/"quarantine.jsonl")
        self.assertEqual(rejected[0]["reason"],"unverified_person_identity")

    def test_split_report_includes_identity_fields(self):
        paths,ws,result=self.prepare_case()
        for key in ("person_disjoint_verified_records","person_unverified_records","person_verified_families","person_unverified_families","verified_person_overlap"):
            self.assertIn(key,result)
