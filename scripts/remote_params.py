"""Count checkpoint parameters without downloading the whole archive: range-read the zip's central
directory and each .safetensors header via the presigned link (link never printed).
Same counting rule as scripts/score.py: sum of all tensor elements in all .safetensors files;
tensors whose key contains visual/vision/vit/image reported separately.
Usage: python remote_params.py <deployment_uuid>"""
import json, ssl, struct, sys, tomllib, urllib.request, re
dep = sys.argv[1]
key = tomllib.load(open(".overmind/credentials.toml", "rb"))["api-key"]
ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")
req = urllib.request.Request(f"https://api.overmindlab.ai/api/deployed-models/{dep}/checkpoints/", headers={"X-Api-Key": key})
meta = json.loads(urllib.request.urlopen(req, timeout=120, context=ctx).read())
f0 = meta["files"][0]; url = f0["download_url"]
print("checkpoint file:", f0["name"], f"{f0['size_bytes']:,} bytes")
def rng(spec):
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={spec}"}), timeout=300, context=ctx)
    return r.read(), r.headers.get("Content-Range")
tail, cr = rng("-262144")
total = int(cr.split("/")[1]); base = total - len(tail)
print(f"archive size: {total:,} bytes ({total/1e9:.2f} GB)")
i = tail.rfind(b"PK\x05\x06")
cd_size, cd_off = struct.unpack("<II", tail[i+12:i+20])
if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
    j = tail.rfind(b"PK\x06\x07", 0, i); eocd64_off = struct.unpack("<Q", tail[j+8:j+16])[0]
    rec = tail[eocd64_off - base: eocd64_off - base + 56]
    assert rec[:4] == b"PK\x06\x06"
    cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
cd = tail[cd_off - base: cd_off - base + cd_size] if cd_off >= base else rng(f"{cd_off}-{cd_off+cd_size-1}")[0]
entries, p = [], 0
while p < len(cd) and cd[p:p+4] == b"PK\x01\x02":
    method, = struct.unpack("<H", cd[p+10:p+12]); csz, usz = struct.unpack("<II", cd[p+20:p+28])
    nl, el, cl = struct.unpack("<HHH", cd[p+28:p+34]); off, = struct.unpack("<I", cd[p+42:p+46])
    name = cd[p+46:p+46+nl].decode(); extra = cd[p+46+nl:p+46+nl+el]
    q = 0
    while q < len(extra):
        hid, hl = struct.unpack("<HH", extra[q:q+4]); body = extra[q+4:q+4+hl]
        if hid == 1:
            vals = list(struct.unpack("<" + "Q" * (hl // 8), body[: hl // 8 * 8])); k = 0
            if usz == 0xFFFFFFFF: usz = vals[k]; k += 1
            if csz == 0xFFFFFFFF: csz = vals[k]; k += 1
            if off == 0xFFFFFFFF: off = vals[k]; k += 1
        q += 4 + hl
    entries.append((name, method, csz, usz, off)); p += 46 + nl + el + cl
print("files in archive:")
for n, m, c, u, o in entries: print(f"  {n}  {u/1e6:,.1f} MB  method={m}")
total_params = vision = 0; dtypes = {}
for n, m, c, u, o in entries:
    if not (n.endswith(".safetensors") or n.endswith("config.json")): continue
    lh, _ = rng(f"{o}-{o+29}"); nl, el = struct.unpack("<HH", lh[26:30]); start = o + 30 + nl + el
    assert m == 0, f"{n} compressed (method {m}); cannot range-read"
    if n.endswith("config.json"):
        cfg = json.loads(rng(f"{start}-{start+u-1}")[0])
        print(f"{n}: model_type={cfg.get('model_type')} archs={cfg.get('architectures')} torch_dtype={cfg.get('torch_dtype') or cfg.get('dtype')} "
              f"hidden={cfg.get('hidden_size')} layers={cfg.get('num_hidden_layers')} vocab={cfg.get('vocab_size')} tie_word_embeddings={cfg.get('tie_word_embeddings')} quant={cfg.get('quantization_config')}")
        continue
    hlen = struct.unpack("<Q", rng(f"{start}-{start+7}")[0])[0]
    hdr = json.loads(rng(f"{start+8}-{start+8+hlen-1}")[0])
    for k, v in hdr.items():
        if k == "__metadata__": continue
        nparam = 1
        for d in v["shape"]: nparam *= d
        total_params += nparam; dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + nparam
        if any(t in k.lower() for t in ("visual", "vision", "vit", "image")): vision += nparam
print(f"total params: {total_params:,} ({total_params/1e9:.4f} B)   dtypes: {dtypes}")
if vision: print(f"  of which vision-tower params: {vision:,} ({vision/1e9:.4f} B)")
