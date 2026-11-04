import io
import unittest
from unittest.mock import patch
from PIL import Image
from tests.support import module,workspace
from tests import test_serving as serving


def envelope(parts,boundary=b"fixture-boundary"):
    chunks=[]
    for name,filename,data in parts:
        disposition=b'Content-Disposition: form-data; name="'+name.encode()+b'"'
        if filename is not None: disposition+=b'; filename="'+filename.encode()+b'"'
        chunks.append(b"--"+boundary+b"\r\n"+disposition+b"\r\n\r\n"+data+b"\r\n")
    return b"".join(chunks)+b"--"+boundary+b"--\r\n"


class BuiltinMultipartTests(unittest.TestCase):
    def test_single_file_matches_raw_api(self):
        api=module(self,"api")
        from fastapi.testclient import TestClient
        ws=module(self,"paths").Workspace(workspace(self))
        stream=io.BytesIO();Image.new("RGB",(100,100),"red").save(stream,format="PNG")
        with patch("app.backend.api.multipart_available",return_value=False):
            app=api.create_app(engine=serving.ServingTests.make_engine(self),workspace=ws)
        with TestClient(app) as client:
            raw=client.post("/analyze",content=stream.getvalue(),headers={"Content-Type":"image/png"})
            multipart=client.post("/analyze",content=envelope([("file","image.png",stream.getvalue())]),headers={"Content-Type":"multipart/form-data; boundary=fixture-boundary"})
        self.assertEqual(multipart.status_code,200)
        self.assertEqual(multipart.json()["faces"],raw.json()["faces"])

    def test_missing_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError,"boundary"):
            module(self,"_multipart").parse_multipart("multipart/form-data",b"invalid")

    def test_oversized_part_rejected(self):
        with self.assertRaisesRegex(ValueError,"part"):
            module(self,"_multipart").parse_multipart("multipart/form-data; boundary=fixture-boundary",envelope([("file","x",b"x"*20)]),max_part_bytes=10)

    def test_malformed_disposition_rejected(self):
        body=envelope([("file","x",b"x")]).replace(b"form-data;",b"attachment;")
        with self.assertRaisesRegex(ValueError,"Disposition"):
            module(self,"_multipart").parse_multipart("multipart/form-data; boundary=fixture-boundary",body)

    def test_multiple_fields_preserve_binary_file(self):
        binary=b"\x00\xff\r\n--fixture-boundary-not-a-delimiter\x80"
        parts=module(self,"_multipart").parse_multipart("multipart/form-data; boundary=fixture-boundary",envelope([("ignored",None,b"value"),("file","x",binary)]))
        self.assertEqual(parts[1]["data"],binary)

    def test_parser_deterministic_and_request_isolated(self):
        api=module(self,"_multipart");header="multipart/form-data; boundary=fixture-boundary"
        body=envelope([("file","x",b"first")]);first=api.parse_multipart(header,body)
        self.assertEqual(first,api.parse_multipart(header,body))
        self.assertEqual(api.parse_multipart(header,envelope([("file","x",b"second")]))[0]["data"],b"second")
        self.assertEqual(first[0]["data"],b"first")
