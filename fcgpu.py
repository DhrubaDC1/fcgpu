#!/usr/bin/env python3
"""fcgpu - export FilmCraft projects on the GPU (VAAPI decode/encode + Vulkan/libplacebo processing).

  fcgpu                                   open the export window
  fcgpu export PROJECT.fcproj OUT.mp4     export from the command line
        [--codec h264|hevc] [--mbps N] [--jobs N] [--check N]

How it stays faithful to FilmCraft: everything FilmCraft computes per pixel (the Lumetri grade, the vignette,
letterbox, titles, audio mix) is rendered BY FilmCraft once per export (a colour-test Hald image, grey and
black/white mattes) and replayed on the GPU as a 3D LUT and overlays. Geometry (scale/position keyframes) and
fades are reproduced from the timeline data. Anything this exporter does not understand stops the export
with a clear message instead of rendering it wrongly. `--check N` compares N frames against FilmCraft's own render.
"""
import argparse, concurrent.futures as cf, hashlib, json, math, os, shutil, subprocess, sys, tempfile, threading, time
from pathlib import Path

T = 254016000000          # FilmCraft ticks per second
RENDER = "/dev/dri/renderD128"
GAMMA = 2.4               # Rec.709 working space: FilmCraft blends/fades in linear light; BT.1886 power (calibrated, see --check)
HALD = 8                  # Hald level 8 -> 64^3 LUT, 512x512 image (100^3 measured no better, and loads slower per clip)
LOG = print


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------- helpers
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd[:6]))} ...\n{(r.stderr or r.stdout)[-2500:]}")
    return r.stdout


def fc_run(project, lines, save=False):
    """Run FilmCraft commands headless on a project; return results (raises on any failed line)."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("".join(json.dumps({"id": i, "params": p}) + "\n" for i, p in lines))
    try:
        args = ["filmcraft-cli", "--compact", "--project", str(project)] + (["--save"] if save else []) + ["run", f.name]
        out = subprocess.run(args, capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    res = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    bad = [r for r in res if not r.get("ok")]
    if bad or len(res) != len(lines):
        raise RuntimeError(f"FilmCraft command failed: {bad[:1] or out.stderr[-1500:]}")
    return [r["result"] for r in res]


def read_rgba16(path):
    import numpy as np
    w, h = map(int, run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)]).split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, "<u2").reshape(h, w, 4).astype(np.float64) / 65535


def write_rgba16(path, a):
    import numpy as np
    h, w = a.shape[:2]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba64le", "-s", f"{w}x{h}", "-i", "-", str(path)],
                   input=np.round(np.clip(a, 0, 1) * 65535).astype("<u2").tobytes(), check=True)


def pval(p):
    """FilmCraft stored param value -> plain python value."""
    (k, v), = p["value"].items()
    return v


# ---------------------------------------------------------------- timeline model
class Timeline:
    def __init__(self, project):
        self.project = Path(project)
        raw = json.load(open(project))["project"]
        self.items = raw["items"]
        seqs = [v for v in self.items.values() if v and "Sequence" in v["kind"]]
        self.insp = json.loads(run(["filmcraft-cli", "--compact", "--project", str(project), "inspect", "sequence"]))
        sid = str(self.insp["id"])
        self.seq = self.items[sid]["kind"]["Sequence"]
        st = self.insp["settings"]
        self.W, self.H = st["width"], st["height"]
        fr = st["frame_rate"]
        self.fps = fr["num"] / fr["den"]
        self.fdur = T * fr["den"] / fr["num"]          # ticks per frame
        self.frames = self.insp["durationFrames"]
        self.raw_clips = {c["id"]: c for tr in self.seq["video_tracks"] for c in tr["items"]}
        self._parse()

    def f(self, ticks):
        return round(ticks / self.fdur)

    def _parse(self):
        vt = self.insp["video"]
        problems = []
        kinds = lambda tr: {list(self.items[str(i["item"])]["kind"])[0] for i in tr["items"]}
        footage = [i for i, tr in enumerate(vt) if tr["items"] and kinds(tr) == {"Media"}]
        adj = [i for i, tr in enumerate(vt) if tr["items"] and kinds(tr) == {"AdjustmentLayer"}]
        gfx = [i for i, tr in enumerate(vt) if tr["items"] and kinds(tr) == {"Graphic"}]
        other = [vt[i]["name"] for i, tr in enumerate(vt) if tr["items"] and i not in footage + adj + gfx]
        if len(footage) != 1: problems.append(f"exactly one video track with footage is supported (found {len(footage)})")
        if len(adj) > 1: problems.append("more than one adjustment-layer track")
        if other: problems.append(f"unsupported track contents on {other}")
        if adj and footage and adj[0] < footage[0]: problems.append("adjustment layer must be above the footage")
        if gfx and adj and gfx[0] < adj[0]: problems.append("titles below the adjustment layer are not supported")
        if problems: raise Unsupported("; ".join(problems))
        self.vtrack = vt[footage[0]]
        self.vtrack_ref = f"V{footage[0] + 1}"
        self.free_track = next((f"V{i + 1}" for i, tr in enumerate(vt) if not tr["items"] and footage[0] < i < (adj[0] if adj else 99)), None)
        if not self.vtrack["enabled"]: raise Unsupported("footage track is disabled")

        # footage clips
        self.clips = []
        for c in sorted(self.vtrack["items"], key=lambda i: i["startFrame"]):
            raw = self.raw_clips[c["clip"]]
            media = self.items[str(c["item"])]["kind"]["Media"]
            path = media["media"]["File"]["path"]
            info = media["info"]["video"]
            if c["speed"] != 1.0 or c["reverse"]: problems.append(f"{c['name']}: speed/reverse")
            if not c["enabled"]: problems.append(f"{c['name']}: disabled clip")
            clip = dict(id=c["clip"], name=c["name"], path=path, a=c["startFrame"], b=c["endFrame"],
                        src_in=c["sourceIn"] / T, iw=info["width"], ih=info["height"], lumetri=None, motion=None)
            for e in raw["effects"]:
                name, ps = e["effect"], e["params"]
                kf = {k for k, p in ps.items() if p.get("keyframes")}
                if not e["enabled"]: continue
                if name == "motion":
                    if kf - {"scale", "position"}: problems.append(f"{c['name']}: animated {kf - {'scale', 'position'}}")
                    if pval(ps["rotation"]) or not pval(ps["uniform_scale"]): problems.append(f"{c['name']}: rotation/non-uniform scale")
                    clip["motion"] = ps
                elif name == "opacity":
                    if kf or pval(ps["opacity"]) != 100 or pval(ps["blend"]) not in (0, "Normal"): problems.append(f"{c['name']}: opacity/blend")
                elif name == "time_remap":
                    if kf: problems.append(f"{c['name']}: time remapping")
                elif name == "lumetri":
                    if kf: problems.append(f"{c['name']}: animated Lumetri")
                    clip["lumetri"] = ps
                else:
                    problems.append(f"{c['name']}: effect '{name}'")
            self.clips.append(clip)
        for x, y in zip(self.clips, self.clips[1:]):
            if x["b"] != y["a"]: problems.append(f"gap/overlap between {x['name']} and {y['name']}")
        if self.clips and (self.clips[0]["a"] != 0 or self.clips[-1]["b"] != self.frames):
            problems.append("footage must cover the whole sequence")

        # fades on the footage track -> per-frame linear-light factor
        self.fade = {}                       # frame -> factor (missing = 1)
        by_id = {c["id"]: c for c in self.clips}
        for tr in self.vtrack["transitions"]:
            if tr["effect"] != "dip_to_black" or tr["reverse"] or tr["params"]:
                problems.append(f"transition '{tr['effect']}' on the footage track"); continue
            s, n = self.f(tr["start"]), self.f(tr["duration"])
            half = n / 2
            for k in range(n):               # first half: picture -> black ; second half: black -> picture
                fr = s + k
                out = (half - k) / half if k < half else (k - half) / half
                if tr["from"] is None and k < half: out = 0.0
                if tr["to"] is None and k >= half: out = 0.0
                self.fade[fr] = min(self.fade.get(fr, 1.0), max(0.0, min(1.0, out)))

        # adjustment layer
        self.adj = None
        if adj:
            items = vt[adj[0]]["items"]
            if len(items) != 1 or items[0]["startFrame"] > 0 or items[0]["endFrame"] < self.frames:
                problems.append("the adjustment layer must be one clip spanning the whole sequence")
            raw = self.raw_clips[items[0]["clip"]]
            for e in raw["effects"]:
                kf = [k for k, p in e["params"].items() if p.get("keyframes")]
                if e["enabled"] and kf: problems.append(f"adjustment layer: animated {e['effect']}")
                if e["enabled"] and e["effect"] not in ("lumetri", "crop", "motion", "opacity", "time_remap"):
                    problems.append(f"adjustment layer: effect '{e['effect']}'")
                if e["effect"] == "motion" and (pval(e["params"]["scale"]) != 100 or pval(e["params"]["rotation"])):
                    problems.append("adjustment layer: motion")
            self.adj = dict(clip=items[0]["clip"], track=f"V{adj[0] + 1}",
                            lumetri_index=next((i for i, e in enumerate(raw["effects"]) if e["effect"] == "lumetri" and e["enabled"]), None))

        # titles / graphics
        self.titles = []
        for gi in gfx:
            tr = vt[gi]
            tmap = {}
            for t in tr["transitions"]:
                if t["effect"] != "cross_dissolve" or (t["from"] is not None and t["to"] is not None):
                    problems.append(f"title transition '{t['effect']}' between clips"); continue
                tmap["in" if t["from"] is None else "out"] = self.f(t["duration"])
            for g in tr["items"]:
                raw = self.raw_clips[g["clip"]]
                for e in raw["effects"]:
                    if any(p.get("keyframes") for p in e["params"].values()): problems.append(f"title '{g['name']}': animation")
                    if e["effect"] == "opacity" and pval(e["params"]["opacity"]) != 100: problems.append("title opacity")
                ins = [t for t in tr["transitions"] if t["to"] == g["clip"]]
                outs = [t for t in tr["transitions"] if t["from"] == g["clip"]]
                self.titles.append(dict(a=g["startFrame"], b=g["endFrame"], track=tr["name"], name=g["name"],
                                        fin=self.f(ins[0]["duration"]) if ins else 0, fout=self.f(outs[0]["duration"]) if outs else 0))
        if problems:
            raise Unsupported("this project uses features the GPU exporter does not support yet:\n  - " + "\n  - ".join(dict.fromkeys(problems)))

    # source window (crop) for a clip as libplacebo expressions in `t` (seconds since the clip's first frame)
    def crop_exprs(self, c):
        motion = next((e for e in self.raw_clips[c["id"]]["effects"] if e["effect"] == "motion"), None)

        def curve(name, comp, default):
            """-> (libplacebo expression in t, python function of t) for one motion component."""
            rp = motion["params"].get(name) if motion else None
            get = lambda v: list(v.values())[0] if comp is None else (list(v.values())[0] or {}).get(comp)
            kfs = (rp or {}).get("keyframes") or []
            if not kfs:
                v = get(rp["value"]) if rp else None
                v = float(default if v is None else v)
                return repr(v), (lambda t: v)
            if any(k["interp"] != "Linear" for k in kfs[:-1]):
                raise Unsupported(f"{c['name']}: only linear keyframes are supported")
            pts = [(k["time"] / T - c["src_in"], float(get(k["value"]))) for k in kfs]
            e = repr(pts[-1][1])
            for (t0, v0), (t1, v1) in reversed(list(zip(pts, pts[1:]))):
                e = f"if(lt(t,{t1:.6f}),{v0}+({v1 - v0})*clip((t-({t0:.6f}))/{max(t1 - t0, 1e-6):.6f},0,1),{e})"
            def fn(t):
                if t <= pts[0][0]: return pts[0][1]
                for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
                    if t < t1: return v0 + (v1 - v0) * min(max((t - t0) / max(t1 - t0, 1e-6), 0), 1)
                return pts[-1][1]
            return f"if(lt(t,{pts[0][0]:.6f}),{pts[0][1]},{e})", fn

        S, s = curve("scale", None, 100.0)
        PX, px = curve("position", "x", self.W / 2)
        PY, py = curve("position", "y", self.H / 2)
        _, ax = curve("anchor", "x", c["iw"] / 2)
        _, ay = curve("anchor", "y", c["ih"] / 2)
        AX, AY = ax(0), ay(0)
        # output pixel X shows source pixel  A + (X - P) * 100/s  -> visible window of the source:
        for t in (0.0, (c["b"] - c["a"] - 1) / self.fps):
            k = 100 / s(t)
            x0, y0, w, h = AX - px(t) * k, AY - py(t) * k, self.W * k, self.H * k
            if x0 < -0.5 or y0 < -0.5 or x0 + w > c["iw"] + 0.5 or y0 + h > c["ih"] + 0.5:
                raise Unsupported(f"{c['name']}: the clip does not fill the frame (black borders are not supported)")
        return dict(crop_x=f"({AX}-({PX})*100/({S}))", crop_y=f"({AY}-({PY})*100/({S}))",
                    crop_w=f"({self.W}*100/({S}))", crop_h=f"({self.H}*100/({S}))")


# ---------------------------------------------------------------- bake (FilmCraft renders the look once)
def bake(tl, work):
    """Render Hald LUT(s), vignette/letterbox overlay and title overlays with FilmCraft itself."""
    import numpy as np
    W, H = tl.W, tl.H
    if HALD ** 2 > min(W, H): raise Unsupported("sequence too small for LUT capture")
    n = HALD * HALD
    k = np.arange(n ** 3)
    hald = np.stack([k % n, (k // n) % n, k // (n * n)], 1) / (n - 1)
    hald_png = work / "hald.png"
    write_rgba16(hald_png, np.concatenate([hald, np.ones((len(hald), 1))], 1).reshape(n * HALD, n * HALD, 4))
    if not tl.free_track:
        raise Unsupported("needs one empty video track between the footage and the adjustment layer (add a track in FilmCraft)")

    tmp = work / "bake.fcproj"
    shutil.copy(tl.project, tmp)
    looks = {}                                   # json(params) -> lut path ; None key = no per-clip lumetri
    for c in tl.clips:
        key = json.dumps(c["lumetri"], sort_keys=True) if c["lumetri"] else ""
        c["lut_key"] = hashlib.sha1(key.encode()).hexdigest()[:10]
        looks[c["lut_key"]] = c["lumetri"]
    mattes = {"g1": "#4d4d4d", "g2": "#b3b3b3", "k": "#000000", "w": "#ffffff"}
    res = fc_run(tmp, [("file.import", {"paths": [str(hald_png)]})] +
                 [("file.newColorMatte", {"color": col, "seconds": 2}) for col in mattes.values()], save=True)
    hald_item = res[0]["items"][0]
    mitem = dict(zip(mattes, (r["item"] for r in res[1:])))

    # times to render at: one second-long slot per pattern, inside the sequence, away from titles
    busy = [(t["a"] - 2, t["b"] + 2) for t in tl.titles]
    slots, f = [], 0
    while len(slots) < len(looks) + 4 + 2 * len(tl.titles):
        if all(not (a <= f + 2 and f <= b) for a, b in busy) and f + 2 < tl.frames: slots.append(f)
        f += 3
        if f >= tl.frames: raise Unsupported("sequence too short for look capture")
    tk = lambda fr: round(fr * tl.fdur)
    ops, outs = [("timeline.setTrack", {"track": tl.vtrack_ref, "enabled": False})], {}
    def shot(fr, name):
        outs[name] = work / f"{name}.png"
        ops.append(("file.exportFrame", {"path": str(outs[name]), "format": "png", "depth": 16, "target": "program", "time": tk(fr) + 1}))
    adjc = tl.adj["clip"] if tl.adj else None
    si = iter(slots)
    if tl.adj:
        for g in ("g1", "g2"):                       # vignette ON
            s = next(si); ops.append(("timeline.place", {"item": mitem[g], "track": tl.free_track, "time": tk(s), "duration": tk(2)})); shot(s, f"{g}_on")
        ops.append(("effects.setParam", {"clip": adjc, "effect": tl.adj["lumetri_index"], "param": "vignette_amount", "value": 0.0})) if tl.adj["lumetri_index"] is not None else None
        for g in ("g1", "g2"):                       # vignette OFF
            s = next(si); ops.append(("timeline.place", {"item": mitem[g], "track": tl.free_track, "time": tk(s), "duration": tk(2)})); shot(s, f"{g}_off")
    for key, lum in looks.items():                   # one LUT per distinct per-clip Lumetri (master grade comes from the adj layer)
        s = next(si)
        ops.append(("timeline.place", {"item": hald_item, "track": tl.free_track, "time": tk(s), "duration": tk(2)}))
        if lum:
            ops.append(("__clip__", None))           # placeholder: the placed clip id is only known at run time
            ops.append(("__lumetri__", (key, lum)))
        shot(s, f"lut_{key}")
    title_slots = []
    if tl.titles:
        if tl.adj: ops.append(("timeline.setTrack", {"track": tl.adj["track"], "enabled": False}))
        for t in tl.titles:
            mid = (t["a"] + t["b"]) // 2
            for g in ("k", "w"):
                ops.append(("timeline.place", {"item": mitem[g], "track": tl.free_track, "time": tk(mid - 1), "duration": tk(3)}))
                shot(mid, f"title{len(title_slots)}_{g}")
            title_slots.append(t)
    # resolve per-clip lumetri: split the batch where a placed clip id is needed
    batch, pending = [], []
    def flush():
        if batch: pending.extend(fc_run(tmp, list(batch), save=True)); batch.clear()
    i = 0
    while i < len(ops):
        op = ops[i]
        if op[0] == "__clip__":
            flush()
            placed = pending[-1]["clips"][0] if isinstance(pending[-1], dict) and "clips" in pending[-1] else None
            key, lum = ops[i + 1][1]
            batch.append(("effects.apply", {"clips": [placed], "effect": "lumetri"}))
            for name, p in lum.items():
                v = pval(p)
                if isinstance(v, dict): v = [v.get("x"), v.get("y")]
                if isinstance(v, list) and len(v) == 4 and all(isinstance(x, float) for x in v):   # colour
                    v = "#%02x%02x%02x" % tuple(round(x * 255) for x in v[:3])
                if v in ("", None): continue
                batch.append(("effects.setParam", {"clip": placed, "effect": "lumetri", "param": name, "value": v}))
            i += 2; continue
        batch.append(op); i += 1
    flush()

    def px(name):
        return read_rgba16(outs[name])
    # LUTs: the hald sits at native size in the frame centre
    y0, x0 = (H - n * HALD) // 2, (W - n * HALD) // 2
    for key in looks:
        img = px(f"lut_{key}")[y0:y0 + n * HALD, x0:x0 + n * HALD, :3].reshape(-1, 3)
        p = work / f"lut_{key}.cube"
        p.write_text(f"LUT_3D_SIZE {n}\n" + "".join(f"{r:.6f} {g:.6f} {b:.6f}\n" for r, g, b in img))
    for c in tl.clips:
        c["lut"] = work / f"lut_{c['lut_key']}.cube"
    # vignette + letterbox overlay: black with alpha so that libplacebo's linear-light blend reproduces the gain
    ov = np.zeros((H, W, 4))
    if tl.adj:
        g = []
        for gname in ("g1", "g2"):
            on, off = px(f"{gname}_on"), px(f"{gname}_off")
            g.append(np.clip(on[..., :3].mean(2) / np.maximum(off[..., :3].mean(2), 1e-6), 0, 1))
            bars = on[..., 3] < 0.5
        gain = (g[0] + g[1]) / 2                      # vignette is ~multiplicative; average the two grey levels
        ov[..., 3] = np.where(bars, 1.0, 1 - gain)    # stage 2 blends on encoded values -> exact encoded gain
    vig_png = work / "frame_overlay.png"
    write_rgba16(vig_png, ov)
    # titles: difference matting over black/white, in linear light
    for i, t in enumerate(title_slots):
        kk, ww = px(f"title{i}_k")[..., :3] ** GAMMA, px(f"title{i}_w")[..., :3] ** GAMMA
        a = np.clip(1 - (ww - kk).mean(2), 0, 1)
        col = np.where(a[..., None] > 1e-4, kk / np.maximum(a[..., None], 1e-4), 0) ** (1 / GAMMA)
        ys, xs = np.where(a > 1e-3)
        if not len(ys): continue
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgba = np.concatenate([np.clip(col, 0, 1), a[..., None]], 2)[y1:y2, x1:x2]
        t["png"] = work / f"title{i}.png"; t["box"] = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        write_rgba16(t["png"], rgba)
    tl.titles = [t for t in title_slots if "png" in t]
    tl.overlay = vig_png


# ---------------------------------------------------------------- render
def segment_cmd(tl, c, out, codec, mbps):
    n = c["b"] - c["a"]
    W, H = tl.W, tl.H
    tags = "color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
    cr = tl.crop_exprs(c)
    inputs = ["-init_hw_device", f"vaapi=va:{RENDER}", "-hwaccel", "vaapi", "-hwaccel_device", "va",
              "-ss", f"{c['src_in']:.6f}", "-i", c["path"]]
    chain = "[0:v]format=yuv420p"
    fades = {fr - c["a"]: v for fr, v in tl.fade.items() if c["a"] <= fr < c["b"] and v < 1}
    if fades:   # linear-light dip to black before the grade: scale encoded values by f^(1/gamma), only on those frames
        lo, hi = min(fades), max(fades)
        e = "1"
        for k in sorted(fades):
            e = f"if(eq(N,{k}),{fades[k] ** (1 / GAMMA):.6f},{e})"
        chain += (f",geq=lum='16+(lum(X,Y)-16)*({e})':cb='128+(cb(X,Y)-128)*({e})':cr='128+(cr(X,Y)-128)*({e})'"
                  f":enable='between(n,{lo},{hi})'")
    chain += (f",libplacebo=w={W}:h={H}:crop_x='{cr['crop_x']}':crop_y='{cr['crop_y']}':crop_w='{cr['crop_w']}':crop_h='{cr['crop_h']}'"
              f":lut='{c['lut']}':lut_type=normalized:{tags}:format=yuv420p10le[g]")
    ov_in = ["-loop", "1", "-framerate", str(tl.fps), "-t", f"{n / tl.fps:.6f}", "-i", str(tl.overlay)]
    lin = tags.replace("color_trc=bt709", "color_trc=linear")
    titles = [t for t in tl.titles if not (t["b"] <= c["a"] or t["a"] >= c["b"])]
    # stage 2: vignette + letterbox. Frames are tagged 'linear' so libplacebo blends the encoded values directly,
    # which reproduces FilmCraft's measured (encoded-space) gain exactly.
    graph = [chain, f"[g]setparams={lin}[gl]", f"[1:v]format=rgba,setparams=color_trc=linear:color_primaries=bt709[ov]",
             f"[gl][ov]libplacebo=inputs=2:w={W}:h={H}:pos_x=0:pos_y=0:pos_w=ow:pos_h=oh:{lin}"
             f":format={'yuv420p10le' if titles else 'nv12'},setparams={tags}[v2]"]
    pins, idx = ["[v2]"], 2
    pos = []
    for t in titles:
        ov_in += ["-loop", "1", "-framerate", str(tl.fps), "-t", f"{n / tl.fps:.6f}", "-i", str(t["png"])]
        # opacity ramp in global frames: in over fin, out over fout (linear opacity, blended in linear light)
        g0 = c["a"]
        op = f"clip(min((N+{g0}-{t['a']})/{max(t['fin'], 1)},({t['b']}-N-{g0})/{max(t['fout'], 1)}),0,1)"
        op = f"if(lt(N+{g0},{t['a']})+gte(N+{g0},{t['b']}),0,{op})"
        graph.append(f"[{idx}:v]format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='alpha(X,Y)*({op})',setparams=color_trc=bt709:color_primaries=bt709[t{idx}]")
        pins.append(f"[t{idx}]"); pos.append((len(pins) - 1, t["box"])); idx += 1
    px = py = pw = ph = None
    if pos:
        def sel(dim, dflt):
            e = dflt
            for i, b in pos:
                e = f"if(eq(in_idx,{i}),{b[dim]},{e})"
            return e
        place = f":pos_x='{sel(0, '0')}':pos_y='{sel(1, '0')}':pos_w='{sel(2, 'ow')}':pos_h='{sel(3, 'oh')}'"
    else:
        place = ""
    if titles:   # stage 3: titles, blended in linear light like FilmCraft
        graph.append(f"{''.join(pins)}libplacebo=inputs={len(pins)}:w={W}:h={H}{place}:{tags}:format=nv12,hwupload[v]")
    else:
        graph[-1] = graph[-1].replace("[v2]", ",hwupload[v]")
    enc = {"h264": ["-c:v", "h264_vaapi", "-profile:v", "high"], "hevc": ["-c:v", "hevc_vaapi", "-profile:v", "main"]}[codec]
    return (["ffmpeg", "-nostdin", "-v", "error", "-y"] + inputs + ov_in +
            ["-filter_complex", ";".join(graph), "-map", "[v]", "-frames:v", str(n), "-filter_hw_device", "va"] + enc +
            ["-rc_mode", "VBR", "-b:v", f"{mbps}M", "-maxrate", f"{mbps * 1.5:.0f}M", "-g", str(round(tl.fps * 2)),
             "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-an", str(out)])


def auto_jobs():
    """Parallel clips by free VRAM: each ffmpeg+libplacebo job peaks at ~1.5 GB for 4K (measured on RX 5500 XT)."""
    try:
        d = Path(RENDER.replace("/dev/dri/renderD128", "/sys/class/drm/card1/device"))
        free = (int((d / "mem_info_vram_total").read_text()) - int((d / "mem_info_vram_used").read_text())) / 2 ** 20
        return int(max(1, min(4, (free - 1000) // 1500)))
    except (OSError, ValueError):
        return 2


def export(project, out, codec="h264", mbps=None, jobs=None, check=0, progress=None):
    project, out = Path(project).resolve(), Path(out).resolve()
    LOG(f"reading {project.name}")
    tl = Timeline(project)
    mbps = mbps or max(8, round(30 * tl.W * tl.H / (3840 * 2160)))
    jobs = jobs or auto_jobs()
    work = Path(tempfile.mkdtemp(prefix="fcgpu-"))
    try:
        LOG("capturing the look from FilmCraft (grade LUTs, vignette/letterbox, titles)")
        t0 = time.time()
        bake(tl, work)
        LOG(f"  ({time.time() - t0:.0f}s)")
        LOG(f"  {len({c['lut_key'] for c in tl.clips})} LUT(s), {len(tl.titles)} title(s)")
        LOG("rendering audio mix (FilmCraft)")
        run(["filmcraft-cli", "--project", str(project), "export", str(work / "audio.wav"), "--format", "wav"])
        LOG(f"rendering {len(tl.clips)} clips, {tl.frames} frames on the GPU ({codec}, {mbps} Mb/s, {jobs} at a time)")
        segs = [work / f"seg{i:04d}.{'mp4'}" for i in range(len(tl.clips))]
        done, lock = [0], threading.Lock()
        def one(i):
            c = tl.clips[i]
            run(segment_cmd(tl, c, segs[i], codec, mbps))
            got = int(run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v", "-show_entries",
                           "stream=nb_read_packets", "-of", "csv=p=0", str(segs[i])]).strip())
            if got != c["b"] - c["a"]:
                raise RuntimeError(f"{c['name']}: rendered {got} frames, expected {c['b'] - c['a']}")
            with lock:
                done[0] += got
                if progress: progress(done[0], tl.frames)
        t0 = time.time()
        with cf.ThreadPoolExecutor(jobs) as ex:
            list(ex.map(one, range(len(segs))))
        LOG(f"  GPU render: {time.time() - t0:.0f}s ({tl.frames / (time.time() - t0):.1f} fps)")
        (work / "list.txt").write_text("".join(f"file '{s}'\n" for s in segs))
        LOG("joining clips + audio")
        dur = tl.frames / tl.fps
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"),
             "-i", str(work / "audio.wav"), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
             "-af", "apad", "-t", f"{dur:.6f}", "-movflags", "+faststart", str(out)])
        LOG(f"done: {out} ({out.stat().st_size / 1e6:.0f} MB)")
        if check:
            compare(tl, out, check, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def compare(tl, out, n, work):
    """Render n frames in FilmCraft and compare with the exported file (mean abs difference, 0-255)."""
    import numpy as np
    picks = sorted(set([tl.frames // 2] + [round(i * (tl.frames - 1) / max(n - 1, 1)) for i in range(n)]))[:n]
    picks += sorted(f for f in tl.fade if 0 < tl.fade[f] < 1)[:2] + [t["a"] + t["fin"] // 2 for t in tl.titles][:1]
    ops = [("file.exportFrame", {"path": str(work / f"ref{f}.png"), "format": "png", "depth": 16, "target": "program",
                                 "time": round(f * tl.fdur) + 1}) for f in picks]
    fc_run(tl.project, ops)
    LOG("check vs FilmCraft's own frame render (mean |difference|, 0-255 levels). H.264 compression alone costs ~2-6;\n"
        "  FilmCraft's own export of the same project measured 2.5-10 on this check:")
    worst = 0
    for f in picks:
        ref = read_rgba16(work / f"ref{f}.png")
        ours = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-vf", f"select=eq(n\\,{f})", "-frames:v", "1",
                               "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], capture_output=True, check=True).stdout
        o = np.frombuffer(ours, "<u2").reshape(tl.H, tl.W, 3).astype(np.float64) / 65535
        r = ref[..., :3] * ref[..., 3:4]                       # transparent (letterbox) = black
        d = np.abs(o - r).mean() * 255
        worst = max(worst, d)
        LOG(f"  frame {f:5d} ({f / tl.fps:6.2f}s): {d:.2f}")
    LOG(f"  worst {worst:.2f}")
    return worst


# ---------------------------------------------------------------- GUI
def gui():
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    root = tk.Tk(); root.title("FilmCraft GPU Export"); root.geometry("620x420")
    proj, outp = tk.StringVar(), tk.StringVar()
    codec, mbps = tk.StringVar(value="h264"), tk.StringVar(value="auto")
    frm = ttk.Frame(root, padding=12); frm.pack(fill="both", expand=True)
    def pick_proj():
        p = filedialog.askopenfilename(title="Open FilmCraft project", filetypes=[("FilmCraft project", "*.fcproj")])
        if p:
            proj.set(p)
            if not outp.get(): outp.set(str(Path(p).with_name(Path(p).stem + "_GPU.mp4")))
    def pick_out():
        p = filedialog.asksaveasfilename(title="Export to", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if p: outp.set(p)
    for r, (label, var, cmd) in enumerate((("Project", proj, pick_proj), ("Export to", outp, pick_out))):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=var, width=52).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(frm, text="Browse…", command=cmd).grid(row=r, column=2)
    ttk.Label(frm, text="Codec").grid(row=2, column=0, sticky="w", pady=4)
    opts = ttk.Frame(frm); opts.grid(row=2, column=1, sticky="w", padx=6)
    ttk.Combobox(opts, textvariable=codec, values=["h264", "hevc"], width=8, state="readonly").pack(side="left")
    ttk.Label(opts, text="   Bitrate (Mb/s)").pack(side="left")
    ttk.Combobox(opts, textvariable=mbps, values=["auto", "20", "30", "45", "60"], width=6).pack(side="left")
    bar = ttk.Progressbar(frm, maximum=1000); bar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)
    log = tk.Text(frm, height=12, state="disabled", wrap="word"); log.grid(row=4, column=0, columnspan=3, sticky="nsew")
    frm.columnconfigure(1, weight=1); frm.rowconfigure(4, weight=1)
    btn = ttk.Button(frm, text="Export with GPU"); btn.grid(row=5, column=2, sticky="e", pady=8)
    def say(msg):
        def f():
            log.configure(state="normal"); log.insert("end", msg + "\n"); log.see("end"); log.configure(state="disabled")
        root.after(0, f)
    def prog(d, total):
        root.after(0, lambda: bar.configure(value=1000 * d / total))
    def work():
        global LOG
        LOG = say
        try:
            export(proj.get(), outp.get(), codec.get(), None if mbps.get() == "auto" else int(mbps.get()), progress=prog)
            root.after(0, lambda: messagebox.showinfo("Export finished", outp.get()))
        except Unsupported as e:
            say(str(e)); root.after(0, lambda: messagebox.showerror("Can't export this project", str(e)))
        except Exception as e:
            say(f"ERROR: {e}"); root.after(0, lambda: messagebox.showerror("Export failed", str(e)[:800]))
        finally:
            root.after(0, lambda: btn.configure(state="normal"))
    def start():
        if not Path(proj.get()).is_file() or not outp.get():
            messagebox.showwarning("FilmCraft GPU Export", "Choose a project and an output file first."); return
        btn.configure(state="disabled"); bar.configure(value=0)
        threading.Thread(target=work, daemon=True).start()
    btn.configure(command=start)
    if len(sys.argv) > 1 and sys.argv[1].endswith(".fcproj"):
        proj.set(sys.argv[1]); outp.set(str(Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem + "_GPU.mp4")))
    root.mainloop()


def main():
    if len(sys.argv) < 2 or sys.argv[1].endswith(".fcproj"):
        return gui()
    ap = argparse.ArgumentParser(prog="fcgpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("project"); e.add_argument("out")
    e.add_argument("--codec", choices=["h264", "hevc"], default="h264")
    e.add_argument("--mbps", type=int)
    e.add_argument("--jobs", type=int, help="clips rendered at once (default: from free GPU memory, max 4)")
    e.add_argument("--check", type=int, default=0, metavar="N", help="compare N frames against FilmCraft's own render")
    a = ap.parse_args()
    last = [-1]
    def prog(d, total):
        pct = 100 * d // total
        if pct != last[0] and pct % 5 == 0:
            last[0] = pct; print(f"  {pct}% ({d}/{total} frames)", flush=True)
    try:
        export(a.project, a.out, a.codec, a.mbps, a.jobs, a.check, prog)
    except Unsupported as err:
        sys.exit(f"fcgpu: {err}")


if __name__ == "__main__":
    main()
