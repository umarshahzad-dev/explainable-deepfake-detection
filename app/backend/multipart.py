"""Bounded binary-safe multipart/form-data parsing using only the standard library."""
from email.message import EmailMessage
from email.parser import BytesParser
from email import policy
import re


def parse_multipart(content_type,body,*,max_bytes=10*1024*1024,max_part_bytes=None,max_parts=16):
    if not isinstance(body,(bytes,bytearray)) or len(body)>max_bytes:
        raise ValueError("Multipart body exceeds byte limit")
    if not isinstance(content_type,str) or "\r" in content_type or "\n" in content_type:
        raise ValueError("Malformed multipart Content-Type boundary")
    header=EmailMessage(policy=policy.default)
    header["Content-Type"]=content_type
    boundary=header.get_boundary()
    params=header.get_params() or []
    if header.get_content_type()!="multipart/form-data" or not boundary or sum(key.lower()=="boundary" for key,value in params)!=1:
        raise ValueError("Multipart boundary is required exactly once")
    if len(boundary)>70 or not re.fullmatch(r"[0-9A-Za-z'()+_,./:=? -]+",boundary) or boundary.endswith(" "):
        raise ValueError("Malformed multipart boundary")
    try: encoded=boundary.encode("ascii");content_type_bytes=content_type.encode("ascii")
    except UnicodeEncodeError as exc: raise ValueError("Multipart boundary/header must be ASCII") from exc
    body=bytes(body)
    delimiter=re.compile(rb"(?:\A|\r\n)--"+re.escape(encoded)+rb"(?:--)?(?:\r\n|\Z)")
    if len(delimiter.findall(body))>max_parts+1:
        raise ValueError("Too many multipart parts")
    if not body.startswith(b"--"+encoded+b"\r\n"):
        raise ValueError("Malformed multipart opening boundary")
    message=BytesParser(policy=policy.default).parsebytes(b"Content-Type: "+content_type_bytes+b"\r\nMIME-Version: 1.0\r\n\r\n"+body)
    if message.defects or not message.is_multipart(): raise ValueError("Malformed multipart boundaries/body")
    parts=[]
    for part in message.iter_parts():
        if len(parts)>=max_parts or part.is_multipart() or part.defects:
            raise ValueError("Malformed or nested multipart part")
        dispositions=part.get_all("Content-Disposition",[])
        if len(dispositions)!=1 or part.get_content_disposition()!="form-data" or getattr(dispositions[0],"defects",()):
            raise ValueError("Malformed Content-Disposition")
        name=part.get_param("name",header="Content-Disposition")
        if not isinstance(name,str) or not name:
            raise ValueError("Content-Disposition requires a field name")
        if len(str(dispositions[0]))>16384: raise ValueError("Multipart part header too large")
        encoding=part.get("Content-Transfer-Encoding","").lower()
        if encoding not in {"","binary","7bit","8bit"}: raise ValueError("Unsupported multipart transfer encoding")
        data=part.get_payload(decode=True)
        if not isinstance(data,bytes): raise ValueError("Malformed binary multipart part")
        if len(data)>(max_part_bytes if max_part_bytes is not None else max_bytes):
            raise ValueError("Multipart part exceeds byte limit")
        filename=part.get_filename()
        if filename is None and len(data)>1024: raise ValueError("Non-file multipart field part exceeds byte limit")
        parts.append({"name":name,"filename":filename,"data":data})
    if not parts: raise ValueError("Empty multipart body")
    return parts
