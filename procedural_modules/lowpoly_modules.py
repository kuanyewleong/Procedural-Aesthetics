import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.spatial import Delaunay
except Exception as e:
    Delaunay = None


# ---------------------------
# Helpers
# ---------------------------

def _to_numpy_img(img01: torch.Tensor) -> np.ndarray:
    """
    img01: torch [3,H,W] in [0,1] -> np [H,W,3] float32
    """
    assert img01.dim() == 3 and img01.shape[0] == 3
    x = img01.detach().float().clamp(0, 1).cpu().numpy()
    return np.transpose(x, (1, 2, 0)).astype(np.float32)

def _to_torch_img(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    img: np [H,W,3] float32 -> torch [3,H,W] in [0,1]
    """
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    t = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    return t.to(device)

def _rgb2gray_np(img: np.ndarray) -> np.ndarray:
    # img: [H,W,3]
    return (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.float32)

def _sobel_edges_np(gray: np.ndarray) -> np.ndarray:
    # simple sobel magnitude (numpy), gray: [H,W]
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # pad reflect
    g = np.pad(gray, ((1,1),(1,1)), mode="reflect")
    gx = (g[:-2,:-2]*kx[0,0] + g[:-2,1:-1]*kx[0,1] + g[:-2,2:]*kx[0,2] +
          g[1:-1,:-2]*kx[1,0] + g[1:-1,1:-1]*kx[1,1] + g[1:-1,2:]*kx[1,2] +
          g[2:,:-2]*kx[2,0] + g[2:,1:-1]*kx[2,1] + g[2:,2:]*kx[2,2])
    gy = (g[:-2,:-2]*ky[0,0] + g[:-2,1:-1]*ky[0,1] + g[:-2,2:]*ky[0,2] +
          g[1:-1,:-2]*ky[1,0] + g[1:-1,1:-1]*ky[1,1] + g[1:-1,2:]*ky[1,2] +
          g[2:,:-2]*ky[2,0] + g[2:,1:-1]*ky[2,1] + g[2:,2:]*ky[2,2])

    mag = np.sqrt(gx*gx + gy*gy + 1e-8)
    return mag.astype(np.float32)

def _add_boundary_points(points: np.ndarray, H: int, W: int, step: int = 32) -> np.ndarray:
    """
    Add boundary + coarse border grid points to stabilize triangulation.
    points: [N,2] (x,y)
    """
    border = []
    # corners
    border += [(0,0), (W-1,0), (0,H-1), (W-1,H-1)]
    # edges at interval
    for x in range(0, W, step):
        border += [(x,0), (x,H-1)]
    for y in range(0, H, step):
        border += [(0,y), (W-1,y)]
    border = np.array(border, dtype=np.int32)
    allp = np.concatenate([points.astype(np.int32), border], axis=0)
    # unique
    allp = np.unique(allp, axis=0)
    return allp

def _add_grid_points(points: np.ndarray, H: int, W: int, grid: int = 64) -> np.ndarray:
    """
    Optional interior grid points -> more uniform triangles.
    grid is approximate spacing in pixels.
    """
    if grid <= 0:
        return points
    xs = np.arange(grid//2, W, grid, dtype=np.int32)
    ys = np.arange(grid//2, H, grid, dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)
    gpts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    allp = np.concatenate([points.astype(np.int32), gpts], axis=0)
    allp = np.unique(allp, axis=0)
    return allp

def _point_in_tri_mask(H: int, W: int, tri: np.ndarray) -> np.ndarray:
    """
    tri: [[x0,y0],[x1,y1],[x2,y2]] int
    Returns boolean mask for pixels inside triangle (barycentric).
    Operates only on triangle bounding box for speed.
    """
    x0,y0 = tri[0]
    x1,y1 = tri[1]
    x2,y2 = tri[2]

    minx = max(int(min(x0,x1,x2)), 0)
    maxx = min(int(max(x0,x1,x2)), W-1)
    miny = max(int(min(y0,y1,y2)), 0)
    maxy = min(int(max(y0,y1,y2)), H-1)
    if maxx < minx or maxy < miny:
        return None, None, None

    xs = np.arange(minx, maxx+1, dtype=np.float32)
    ys = np.arange(miny, maxy+1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # barycentric coordinates
    den = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
    if abs(den) < 1e-8:
        return None, None, None

    a = ((y1 - y2)*(X - x2) + (x2 - x1)*(Y - y2)) / den
    b = ((y2 - y0)*(X - x2) + (x0 - x2)*(Y - y2)) / den
    c = 1.0 - a - b

    mask = (a >= 0) & (b >= 0) & (c >= 0)
    return mask, (miny, maxy, minx, maxx), (Y, X)  # coords not always needed


# ---------------------------
# Modules
# ---------------------------

class FeaturePointsEdges:
    """
    Feature points from strong gradients (Sobel edges) + random sampling on edges.
    This is robust and dependency-light (no skimage required).
    """
    def __init__(self, num_points: int = 600, edge_quantile: float = 0.90, jitter: int = 2, seed: int = 0):
        self.num_points = int(num_points)
        self.edge_quantile = float(edge_quantile)
        self.jitter = int(jitter)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> np.ndarray:
        img = _to_numpy_img(img01)
        H, W = img.shape[:2]
        gray = _rgb2gray_np(img)
        edges = _sobel_edges_np(gray)
        thr = np.quantile(edges, self.edge_quantile)
        ys, xs = np.where(edges >= thr)

        if len(xs) == 0:
            # fallback: uniform random points
            xs = self.rng.integers(0, W, size=self.num_points)
            ys = self.rng.integers(0, H, size=self.num_points)
        else:
            idx = self.rng.choice(len(xs), size=min(self.num_points, len(xs)), replace=False)
            xs = xs[idx]
            ys = ys[idx]

        pts = np.stack([xs, ys], axis=1).astype(np.int32)

        # small jitter to avoid degenerate triangulations
        if self.jitter > 0:
            jx = self.rng.integers(-self.jitter, self.jitter+1, size=pts.shape[0])
            jy = self.rng.integers(-self.jitter, self.jitter+1, size=pts.shape[0])
            pts[:, 0] = np.clip(pts[:, 0] + jx, 0, W-1)
            pts[:, 1] = np.clip(pts[:, 1] + jy, 0, H-1)

        pts = np.unique(pts, axis=0)
        return pts


class DelaunayTriangulator:
    """
    Builds Delaunay triangulation from points.
    Adds boundary and optional interior grid points.
    """
    def __init__(self, boundary_step: int = 32, grid_step: int = 0):
        self.boundary_step = int(boundary_step)
        self.grid_step = int(grid_step)

    @torch.no_grad()
    def __call__(self, points_xy: np.ndarray, H: int, W: int):
        if Delaunay is None:
            raise RuntimeError("scipy is required for Delaunay triangulation. `pip install scipy`")

        pts = points_xy.astype(np.int32)
        pts = _add_boundary_points(pts, H, W, step=self.boundary_step)
        pts = _add_grid_points(pts, H, W, grid=self.grid_step)

        tri = Delaunay(pts.astype(np.float64))
        # tri.simplices are indices into pts, shape [T,3]
        return pts, tri.simplices.astype(np.int32)


class FlatShadeTriangles:
    """
    Rasterizes triangles and fills each with mean color from original image.
    """
    def __init__(self, edge_overlay: bool = False, edge_strength: float = 0.35):
        self.edge_overlay = bool(edge_overlay)
        self.edge_strength = float(edge_strength)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor, pts: np.ndarray, simplices: np.ndarray) -> torch.Tensor:
        device = img01.device
        img = _to_numpy_img(img01)  # [H,W,3]
        H, W = img.shape[:2]
        out = np.zeros_like(img, dtype=np.float32)

        # optional edge overlay buffer
        edge_mask = np.zeros((H, W), dtype=np.float32) if self.edge_overlay else None

        for s in simplices:
            tri = pts[s]  # [[x,y],[x,y],[x,y]]
            mask, bbox, _ = _point_in_tri_mask(H, W, tri)
            if mask is None:
                continue
            miny, maxy, minx, maxx = bbox
            region = img[miny:maxy+1, minx:maxx+1, :]
            if region.size == 0:
                continue
            m = mask
            if m.sum() == 0:
                continue
            mean_col = region[m].mean(axis=0)  # [3]
            out_reg = out[miny:maxy+1, minx:maxx+1, :]
            out_reg[m] = mean_col
            out[miny:maxy+1, minx:maxx+1, :] = out_reg

            if edge_mask is not None:
                # draw edges by marking boundary pixels of triangle bbox mask
                # simple approx: mark pixels where mask changes (cheap)
                mm = m.astype(np.float32)
                gx = np.abs(mm[:, 1:] - mm[:, :-1]).sum()
                gy = np.abs(mm[1:, :] - mm[:-1, :]).sum()
                # if there is an edge, mark a thin border by dilation of boundary
                # quick boundary detection:
                bd = np.zeros_like(mm, dtype=np.float32)
                bd[:, 1:] = np.maximum(bd[:, 1:], np.abs(mm[:, 1:] - mm[:, :-1]))
                bd[1:, :] = np.maximum(bd[1:, :], np.abs(mm[1:, :] - mm[:-1, :]))
                edge_mask[miny:maxy+1, minx:maxx+1] = np.maximum(
                    edge_mask[miny:maxy+1, minx:maxx+1], bd
                )

        if edge_mask is not None:
            # darken along edges
            out = out * (1.0 - self.edge_strength * edge_mask[..., None])

        return _to_torch_img(out, device=device)


class LowPolyPipeline:
    """
    Full low-poly pipeline:
      feature points -> delaunay -> flat shading -> optional edge overlay
    """
    def __init__(
        self,
        num_points: int = 600,
        edge_quantile: float = 0.90,
        boundary_step: int = 32,
        grid_step: int = 0,           # 0 disables interior grid points
        edge_overlay: bool = True,
        edge_strength: float = 0.35,
        jitter: int = 2,
        seed: int = 0,
    ):
        self.fp = FeaturePointsEdges(
            num_points=num_points,
            edge_quantile=edge_quantile,
            jitter=jitter,
            seed=seed
        )
        self.tri = DelaunayTriangulator(boundary_step=boundary_step, grid_step=grid_step)
        self.shade = FlatShadeTriangles(edge_overlay=edge_overlay, edge_strength=edge_strength)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        img = _to_numpy_img(img01)
        H, W = img.shape[:2]
        pts = self.fp(img01)                     # [N,2] x,y
        pts, simplices = self.tri(pts, H, W)     # triangulate
        return self.shade(img01, pts, simplices).clamp(0, 1)
