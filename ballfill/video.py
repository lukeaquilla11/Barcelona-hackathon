"""Ball detection in the tactical video, trained on our own tracking (no hand labels).

The tactical homography maps the pitch to the image: pixel = H @ (x / 105.3, y / 68, 1). Row i of
the homography is frame i of the video, which is frame i of the tracking. So every frame where both
cameras agree on the ball is a labelled example of what the ball looks like in this video.

1. Training set: crops of three consecutive frames (f-1, f, f+1) around the projected ball, from
   frames where both trackers agree (source "agree", no edge inflation). Negatives are crops away
   from the ball, half of them at the feet of players who don't have it.
2. A small heatmap network (stride-2 stem, U-Net) is trained on one half and applied to the other,
   so every detection is out of sample for test sets A and B.
3. The projected tracker position is ~10 px from the ball's centre (tracking error, and the ball's
   centre sits above the ground point). Round 1 trains with a wide target; its peak near each label
   becomes the label for round 2, with a tight target. Labels it can't find are dropped.
4. Detection runs on every graded frame, full frame, and keeps the top peaks per frame.

    python -m ballfill.video extract   # training crops, both halves (~10 min)
    python -m ballfill.video train     # two rounds, both halves (~40 min)
    python -m ballfill.video detect    # every graded frame (~80 min)
"""

import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, require  # noqa: E402
from possession.load_data import CACHE_DIR  # noqa: E402

L, W = 105.3, 68.0
IMG_W, IMG_H, PAD_H = 1920, 1080, 1088
CROP = 128
N_POS, N_NEG = 10000, 6000
SIGMA_PX = {1: 8.0, 2: 3.0, 3: 3.0}       # target Gaussian, full-res pixels, by round
REFINE_PX = 24                     # round-1 peak must be this close to the projected label
REFINE_MIN = 0.3
TOP_K = 5
MIN_PEAK = 0.02


def video_path(game=GAME):
    """data/tactical/<game>_tactical.mp4 as games.py lays it out, or loose in data/."""
    try:
        return require(["video"], game)[0]
    except Exception:
        data_dir = os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
        p = os.path.join(data_dir, "{}_tactical.mp4".format(game))
        if not os.path.exists(p):
            raise
        return p


def to_pixel(H, xy):
    """(..., 2) metres -> (..., 2) pixels. H (..., 3, 3)."""
    p = np.stack([xy[..., 0] / L, xy[..., 1] / W, np.ones(xy.shape[:-1])], -1)
    v = np.einsum("...ij,...j->...i", H, p)
    return v[..., :2] / v[..., 2:3]


def to_pitch(H, uv):
    """(..., 2) pixels -> (..., 2) metres on the ground plane."""
    p = np.stack([uv[..., 0], uv[..., 1], np.ones(uv.shape[:-1])], -1)
    v = np.einsum("...ij,...j->...i", np.linalg.inv(H), p)
    n = v[..., :2] / v[..., 2:3]
    return np.stack([n[..., 0] * L, n[..., 1] * W], -1)


def half_frames(match, half):
    idx = np.flatnonzero(match.half == half)
    return int(idx[0]), int(idx[-1]) + 1


PREFIX = "ballvid_"       # ballfill.bvideo switches this to keep the broadcast files apart


def cache(name):
    return os.path.join(CACHE_DIR, PREFIX + name)


# --------------------------------------------------------------------------------------------
# training set

def _choose(d, half, seed=0):
    """Positive frames + ball pixel, negative frames + crop centre."""
    from ballfill.fuse import fuse
    m = d.match
    ball, src, sig, _ = fuse(d)
    a, b = half_frames(m, half)
    H = m.homography.astype(np.float64)
    rng = np.random.default_rng(seed + half)
    f = np.arange(a + 1, b - 1)
    ok = (src[f] == 1) & (sig[f] <= 0.45)
    f = f[ok]
    uv = to_pixel(H[f], ball[f])
    inside = (uv[:, 0] > 8) & (uv[:, 0] < IMG_W - 8) & (uv[:, 1] > 8) & (uv[:, 1] < IMG_H - 8)
    f, uv = f[inside], uv[inside]
    pick = np.sort(rng.choice(len(f), min(N_POS, len(f)), replace=False))
    pos_f, pos_uv = f[pick], uv[pick]
    # negatives: other agree frames; half at the feet of a player >= 4 m from the ball
    pick = np.sort(rng.choice(len(f), N_NEG, replace=False))
    neg_f, neg_c = [], []
    for k in pick:
        fr, buv = f[k], uv[k]
        for _ in range(20):
            if rng.random() < 0.5:
                p = m.players_xy[fr]
                far = np.isfinite(p[:, 0]) & (np.linalg.norm(p - ball[fr], axis=1) > 4)
                if not far.any():
                    continue
                c = to_pixel(H[fr], p[rng.choice(np.flatnonzero(far))]) + rng.normal(0, 20, 2)
            else:
                c = np.array([rng.uniform(0, IMG_W), rng.uniform(0, IMG_H)])
            c = np.clip(c, CROP / 2, [IMG_W - CROP / 2, IMG_H - CROP / 2])
            if np.all(np.abs(buv - c) > CROP / 2 + 8):
                neg_f.append(fr)
                neg_c.append(c)
                break
    return pos_f, pos_uv, np.array(neg_f), np.array(neg_c)


def _choose_r3(d, half, n_pos=12000, n_hard=8000, n_rand=4000, seed=0):
    """Round 3, from the round-2 detections on this half (made by the other half's network, so
    nothing about the half it is later applied to is used):
    - positives: agreed frames where a peak (score >= 0.3) lies within 1.5 m of the agreed ball;
      the label is that peak's pixel, so labels are on the ball rather than ~10 px off.
    - hard negatives: peaks with score >= 0.3 more than 5 m from the agreed ball and more than
      2 m off the camera ray through it (a ball in the air projects along that ray).
    - random negatives as in round 1."""
    from ballfill.fuse import CAMERA_XY, fuse
    m = d.match
    ball, src, sig, _ = fuse(d)
    a, b = half_frames(m, half)
    H = m.homography.astype(np.float64)
    r2 = cache("det_h{}_r2.npz".format(half))           # round-2 detections (renamed after round 3)
    z = np.load(r2 if os.path.exists(r2) else cache("det_h{}.npz".format(half)))
    det = np.full((d.n, TOP_K, 3), np.nan, dtype=np.float32)
    det[int(z["first"]):int(z["first"]) + len(z["peaks"])] = z["peaks"]
    rng = np.random.default_rng(seed + 10 * half)
    f = np.arange(a + 1, b - 1)
    f = f[(src[f] == 1) & (sig[f] <= 0.45)]
    xy = to_pitch(H[f][:, None], det[f, :, :2].astype(np.float64))
    s = np.nan_to_num(det[f, :, 2], nan=0.0)
    dist = np.linalg.norm(xy - ball[f][:, None], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    near = (dist <= 1.5) & (s >= 0.3)
    j = np.where(near, s, -1).argmax(1)
    okp = near.any(1)
    pf, puv = f[okp], det[f[okp], j[okp], :2].astype(np.float64)
    pick = np.sort(rng.choice(len(pf), min(n_pos, len(pf)), replace=False))
    pos_f, pos_uv = pf[pick], puv[pick]
    # off the camera ray: perpendicular distance from the line camera -> agreed ball
    ray = ball[f] - CAMERA_XY
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    rel = xy - CAMERA_XY
    perp = np.abs(rel[..., 0] * ray[:, None, 1] - rel[..., 1] * ray[:, None, 0])
    hard = (s >= 0.3) & (dist > 5) & (perp > 2)
    hi, hk = np.nonzero(hard)
    pick = rng.choice(len(hi), min(n_hard, len(hi)), replace=False)
    neg_f = list(f[hi[pick]])
    neg_c = list(det[f[hi[pick]], hk[pick], :2].astype(np.float64) + rng.normal(0, 12, (len(pick), 2)))
    _, _, rf, rc = _choose(d, half, seed=seed + 7)
    pick = rng.choice(len(rf), min(n_rand, len(rf)), replace=False)
    neg_f += list(rf[pick])
    neg_c += list(rc[pick])
    neg_c = np.clip(np.array(neg_c), CROP / 2, [IMG_W - CROP / 2, IMG_H - CROP / 2])
    print("half {}: {} clean positives available, {} hard negatives available".format(
        half, len(pf), len(hi)))
    return pos_f, pos_uv, np.array(neg_f), neg_c


def extract(d, half, rnd=1, chosen=None, path=None):
    """Write cache/ballvid_train_h{half}_X.npy (N, 9, CROP, CROP) uint8 and _meta.npz.

    chosen: (pos_f, pos_uv, neg_f, neg_c) on the frames of the video at `path` (default: the
    tactical video, examples chosen by round)."""
    import cv2
    pos_f, pos_uv, neg_f, neg_c = chosen if chosen is not None else \
        (_choose if rnd == 1 else _choose_r3)(d, half)
    rng = np.random.default_rng(100 + half)
    # crop origin: positives with the ball at a random place inside the crop
    pos_o = np.floor(pos_uv - rng.uniform(16, CROP - 16, pos_uv.shape)).astype(int)
    neg_o = np.floor(neg_c - CROP / 2).astype(int)
    frames = np.r_[pos_f, neg_f]
    origin = np.clip(np.r_[pos_o, neg_o], 0, [IMG_W - CROP, IMG_H - CROP])
    label = np.r_[pos_uv, np.full((len(neg_f), 2), np.nan)] - origin
    label[len(pos_f):] = -1
    order = np.argsort(frames, kind="stable")
    frames, origin, label = frames[order], origin[order], label[order]
    n = len(frames)
    X = np.lib.format.open_memmap(cache("train_h{}_X.npy".format(half)), mode="w+",
                                  dtype=np.uint8, shape=(n, 9, CROP, CROP))
    need = set()
    for fr in frames:
        need.update((fr - 1, fr, fr + 1))
    cap = cv2.VideoCapture(path or video_path())
    first, last = frames[0] - 1, frames[-1] + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    buf, j, t0 = {}, 0, time.time()
    for fr in range(first, last + 1):
        if not cap.grab():
            raise RuntimeError("video ended at frame {}".format(fr))
        if fr in need:
            buf[fr] = cap.retrieve()[1]
        while j < n and frames[j] + 1 <= fr:
            c = frames[j]
            x0, y0 = origin[j]
            X[j] = np.concatenate([buf[c + k][y0:y0 + CROP, x0:x0 + CROP].transpose(2, 0, 1)
                                   for k in (-1, 0, 1)])
            j += 1
        for k in [k for k in buf if k < fr - 2]:
            del buf[k]
    X.flush()
    np.savez(cache("train_h{}_meta.npz".format(half)), frames=frames, origin=origin, label=label)
    print("half {}: {} crops ({} positive) in {:.0f} s".format(half, n, len(pos_f),
                                                               time.time() - t0))


# --------------------------------------------------------------------------------------------
# network

def make_net():
    import torch.nn as nn
    import torch

    def blk(i, o, s=1):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.BatchNorm2d(o), nn.ReLU())

    class Net(nn.Module):
        """Heatmap at half resolution: output pixel (i, j) is input pixel (2i, 2j)."""

        def __init__(self, c=(16, 32, 64)):
            super().__init__()
            a, b, c3 = c
            self.stem = blk(9, a, 2)
            self.e1 = blk(a, a)
            self.e2 = nn.Sequential(blk(a, b, 2), blk(b, b))
            self.e3 = nn.Sequential(blk(b, c3, 2), blk(c3, c3), blk(c3, c3))
            self.u2 = nn.Sequential(nn.Upsample(scale_factor=2), blk(c3, b))
            self.d2 = blk(2 * b, b)
            self.u1 = nn.Sequential(nn.Upsample(scale_factor=2), blk(b, a))
            self.d1 = blk(2 * a, a)
            self.out = nn.Conv2d(a, 1, 1)
            self.out.bias.data.fill_(-4.0)

        def forward(self, x):
            a = self.e1(self.stem(x))
            b = self.e2(a)
            c = self.e3(b)
            b2 = self.d2(torch.cat([self.u2(c), b], 1))
            a2 = self.d1(torch.cat([self.u1(b2), a], 1))
            return self.out(a2)

    return Net()


def _targets(label, sigma_px, size):
    """(B, 2) crop-pixel labels (-1 = none) -> (B, 1, size, size) Gaussian heatmaps."""
    import torch
    g = (torch.arange(size, device=label.device, dtype=torch.float32) * 2.0)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    d2 = (xx[None] - label[:, 0, None, None]) ** 2 + (yy[None] - label[:, 1, None, None]) ** 2
    t = torch.exp(-d2 / (2 * sigma_px ** 2))
    t[label[:, 0] < 0] = 0
    return t[:, None]


def focal_loss(logit, target):
    """CenterNet penalty-reduced focal loss."""
    import torch
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = target > 0.99
    neg_w = (1 - target) ** 4
    lp = -(torch.log(p) * (1 - p) ** 2)[pos].sum()
    ln = -(torch.log(1 - p) * p ** 2 * neg_w)[~pos].sum()
    return (lp + ln) / max(int(pos.sum()), 1)


def _batches(X, label, idx, bs, rng, augment):
    import torch
    for s in range(0, len(idx), bs):
        j = np.sort(idx[s:s + bs])
        x = torch.from_numpy(np.ascontiguousarray(X[j])).float() / 255.0 - 0.5
        y = torch.from_numpy(label[j].astype(np.float32))
        if augment:
            flip = torch.from_numpy(rng.random(len(j)) < 0.5)
            x[flip] = x[flip].flip(-1)
            y[flip, 0] = torch.where(y[flip, 0] >= 0, CROP - 1 - y[flip, 0], y[flip, 0])
            gain = torch.from_numpy(rng.uniform(0.8, 1.2, (len(j), 1, 1, 1)).astype(np.float32))
            bias = torch.from_numpy(rng.uniform(-0.08, 0.08, (len(j), 1, 1, 1)).astype(np.float32))
            x = (x + 0.5) * gain - 0.5 + bias
        yield x, y


def train_round(half, rnd, label, epochs=14, bs=64, seed=0):
    import torch
    dev = "mps"
    X = np.load(cache("train_h{}_X.npy".format(half)), mmap_mode="r")
    keep = np.flatnonzero(~np.isnan(label[:, 0]))
    rng = np.random.default_rng(seed)
    net = make_net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(keep) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        tot, nb = 0.0, 0
        for x, y in _batches(X, label, rng.permutation(keep), bs, rng, True):
            x, y = x.to(dev), y.to(dev)
            loss = focal_loss(net(x), _targets(y, SIGMA_PX[rnd], CROP // 2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        print("  half {} round {} epoch {:2d} loss {:.4f} ({:.0f} s)".format(
            half, rnd, ep, tot / nb, time.time() - t0), flush=True)
    torch.save(net.state_dict(), cache("net_h{}_r{}.pt".format(half, rnd)))
    return net


def predict_crops(net, half):
    """Peak (u, v, score) per training crop, crop pixels."""
    import torch
    X = np.load(cache("train_h{}_X.npy".format(half)), mmap_mode="r")
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 256):
            x = torch.from_numpy(np.ascontiguousarray(X[s:s + 256])).float().to("mps") / 255 - 0.5
            h = torch.sigmoid(net(x))[:, 0]
            out.append(h.cpu().numpy())
    return np.concatenate(out)


def refine(heat, label):
    """Round-2 labels: round-1 peak within REFINE_PX of each positive label, else NaN (dropped)."""
    size = heat.shape[-1]
    g = np.arange(size) * 2.0
    new = label.copy()
    for i in np.flatnonzero(label[:, 0] >= 0):
        h = heat[i]
        d2 = (g[None, :] - label[i, 0]) ** 2 + (g[:, None] - label[i, 1]) ** 2
        hm = np.where(d2 <= REFINE_PX ** 2, h, 0)
        r, c = np.unravel_index(np.argmax(hm), hm.shape)
        if hm[r, c] < REFINE_MIN:
            new[i] = np.nan
            continue
        r0, r1, c0, c1 = max(r - 1, 0), min(r + 2, size), max(c - 1, 0), min(c + 2, size)
        w = h[r0:r1, c0:c1]
        rr, cc = np.mgrid[r0:r1, c0:c1]
        new[i] = [2.0 * (w * cc).sum() / w.sum(), 2.0 * (w * rr).sum() / w.sum()]
    return new


def train(half):
    meta = np.load(cache("train_h{}_meta.npz".format(half)))
    label = meta["label"].astype(np.float64)
    net = train_round(half, 1, label)
    heat = predict_crops(net, half)
    new = refine(heat, label)
    pos = label[:, 0] >= 0
    kept = pos & ~np.isnan(new[:, 0])
    shift = np.linalg.norm(new[kept] - label[kept], axis=1)
    print("half {}: refined {:.1%} of positives, label shift median {:.1f} px".format(
        half, kept.sum() / pos.sum(), np.median(shift)))
    np.save(cache("train_h{}_label_r2.npy".format(half)), new)
    train_round(half, 2, new)


# --------------------------------------------------------------------------------------------
# detection

def load_net(half, rnd=2):
    import torch
    net = make_net()
    net.load_state_dict(torch.load(cache("net_h{}_r{}.pt".format(half, rnd)), map_location="cpu"))
    return net.to("mps").eval().half()


def _reader(path, first, last, q):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for fr in range(first, last + 1):
        ok, img = cap.read()
        q.put((fr, img if ok else None))
    q.put(None)


def peaks(heat, k=TOP_K):
    """(B, h, w) sigmoid heatmaps -> (B, k, 3) full-res (u, v, score), NaN where none."""
    import torch
    import torch.nn.functional as F
    mx = F.max_pool2d(heat[:, None], 7, 1, 3)[:, 0]
    nms = torch.where(heat == mx, heat, torch.zeros_like(heat))
    B, h, w = heat.shape
    s, i = nms.view(B, -1).topk(k)
    r, c = (i // w).float(), (i % w).float()
    # sub-pixel: 3x3 weighted centroid
    pad = F.pad(heat[:, None], (1, 1, 1, 1))[:, 0]
    du = torch.zeros_like(s)
    dv = torch.zeros_like(s)
    ws = torch.zeros_like(s)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            ii = ((r.long() + 1 + dr) * (w + 2) + (c.long() + 1 + dc))
            v = pad.reshape(B, -1).gather(1, ii)
            du += v * dc
            dv += v * dr
            ws += v
    u = 2.0 * (c + du / ws)
    v = 2.0 * (r + dv / ws)
    out = torch.stack([u, v, s], -1)
    out[s < MIN_PEAK] = float("nan")
    return out


def detect(match, half, model_half, bs=4, rnd=2):
    """Top peaks on every frame of `half`, with the network trained on `model_half`."""
    import torch
    net = load_net(model_half, rnd)
    a, b = half_frames(match, half)
    res = np.full((b - a, TOP_K, 3), np.nan, dtype=np.float32)
    q = queue.Queue(maxsize=32)
    th = threading.Thread(target=_reader, args=(video_path(), a - 1, b, q), daemon=True)
    th.start()
    ring, batch, bframes, t0 = {}, [], [], time.time()

    def flush():
        x = torch.stack(batch)
        with torch.no_grad():
            h = torch.sigmoid(net(x).float())[:, 0]
        p = peaks(h).cpu().numpy()
        for fr, pk in zip(bframes, p):
            res[fr - a] = pk
        batch.clear()
        bframes.clear()

    while True:
        item = q.get()
        if item is None:
            break
        fr, img = item
        if img is None:
            ring[fr] = None
        else:
            t = torch.from_numpy(img).to("mps").permute(2, 0, 1).half() / 255 - 0.5
            ring[fr] = torch.nn.functional.pad(t, (0, 0, 0, PAD_H - IMG_H))
        c = fr - 1
        if a <= c < b and all(ring.get(c + k) is not None for k in (-1, 0, 1)):
            batch.append(torch.cat([ring[c - 1], ring[c], ring[c + 1]]))
            bframes.append(c)
            if len(batch) == bs:
                flush()
        ring.pop(fr - 3, None)
        if (fr - a) % 5000 == 0:
            done = fr - a + 1
            print("  half {}: {}/{} frames, {:.1f} fps".format(
                half, done, b - a, done / (time.time() - t0)), flush=True)
    if batch:
        flush()
    np.savez(cache("det_h{}.npz".format(half)), first=a, peaks=res, model_half=model_half)
    print("half {} done in {:.0f} s".format(half, time.time() - t0))


def load_detections(n, suffix=None):
    """(n, TOP_K, 3) pixel peaks for every frame (NaN outside the graded halves).

    suffix picks an older round's file (e.g. "_r2"); default $BALLVID_DET, else the latest."""
    suffix = os.environ.get("BALLVID_DET", "") if suffix is None else suffix
    out = np.full((n, TOP_K, 3), np.nan, dtype=np.float32)
    for half in (1, 2):
        z = np.load(cache("det_h{}{}.npz".format(half, suffix)))
        a = int(z["first"])
        out[a:a + len(z["peaks"])] = z["peaks"]
    return out


def main(argv):
    from ballfill.data import load_ball_data
    step = argv[1] if len(argv) > 1 else "all"
    d = load_ball_data()
    if step in ("extract", "all"):
        for h in (1, 2):
            extract(d, h)
    if step in ("train", "all"):
        for h in (1, 2):
            train(h)
    if step in ("detect", "all"):
        for h in (1, 2):
            detect(d.match, h, model_half=3 - h)
    if step == "round3":
        # clean labels + hard negatives mined from the round-2 detections, one tight round
        for h in (1, 2):
            extract(d, h, rnd=3)
            meta = np.load(cache("train_h{}_meta.npz".format(h)))
            train_round(h, 3, meta["label"].astype(np.float64), epochs=16)
        for h in (1, 2):
            os.replace(cache("det_h{}.npz".format(h)), cache("det_h{}_r2.npz".format(h)))
        for h in (1, 2):
            detect(d.match, h, model_half=3 - h, rnd=3)


if __name__ == "__main__":
    main(sys.argv)

