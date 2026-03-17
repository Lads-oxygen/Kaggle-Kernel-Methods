import numpy as np


class DenseSIFT:
    """Dense SIFT-like descriptor on a fixed grid.

    This implementation is designed for small images represented as flattened
    vectors in CHW order (e.g., 3x32x32). It computes gradient-orientation
    histograms over cells in each patch, applies an L2 normalisation + clipping
    scheme, and concatenates descriptors across the grid into one fixed-length
    vector per image.

    Notes:
        - This is a SIFT-style histogram descriptor; it is not a full keypoint
          detector.
        - Output features are non-negative and suited to histogram kernels.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        colour_mode: str = "rgb",
        patch_size: int = 16,
        stride: int = 8,
        num_cells: int = 4,
        num_angles: int = 8,
        clip: float = 0.2,
        eps: float = 1e-12,
    ):
        self.image_shape = tuple(int(x) for x in image_shape)
        self.colour_mode = str(colour_mode)
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.num_cells = int(num_cells)
        self.num_angles = int(num_angles)
        self.clip = float(clip)
        self.eps = float(eps)

        c, h, w = self.image_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError("image_shape must be positive")
        if self.patch_size <= 0 or self.stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        if self.patch_size > min(h, w):
            raise ValueError("patch_size cannot exceed image dimensions")
        if self.patch_size % self.num_cells != 0:
            raise ValueError("patch_size must be divisible by num_cells")
        if self.num_angles <= 0:
            raise ValueError("num_angles must be positive")

        if self.colour_mode not in {"rgb", "gray"}:
            raise ValueError("colour_mode must be 'rgb' or 'gray'")

        self._cell_size = self.patch_size // self.num_cells

    def fit(self, X: np.ndarray) -> "DenseSIFT":
        """No-op fit for transformer-style compatibility."""

        _ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute dense SIFT-like features for a batch.

        Args:
            X: Array of shape (n, C*H*W) or a single sample of shape (C*H*W,).

        Returns:
            Feature matrix of shape (n, d_out) (or (d_out,) for single input).
        """

        X_arr = np.asarray(X)
        single = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            single = True
        if X_arr.ndim != 2:
            raise ValueError("X must be 1D or 2D")

        c, h, w = self.image_shape
        expected = c * h * w
        if X_arr.shape[1] != expected:
            raise ValueError(f"Expected vectors of length {expected}, got {X_arr.shape[1]}")

        feats = [self._transform_one(X_arr[i]) for i in range(X_arr.shape[0])]
        F = np.stack(feats, axis=0)
        return F[0] if single else F

    def _transform_one(self, v: np.ndarray) -> np.ndarray:
        c, h, w = self.image_shape
        img = np.asarray(v, dtype=np.float32).reshape(c, h, w)

        ps = self.patch_size
        st = self.stride
        ys = list(range(0, h - ps + 1, st))
        xs = list(range(0, w - ps + 1, st))

        if c == 1 or self.colour_mode == "gray":
            gray = img.mean(axis=0) if c > 1 else img[0]
            mag, ori = self._gradients(gray)

            descs = []
            for y0 in ys:
                for x0 in xs:
                    descs.append(self._patch_descriptor(mag, ori, y0, x0))
            return np.concatenate(descs, axis=0).astype(np.float32, copy=False)

        # Per-channel descriptors (e.g., RGB) concatenated.
        mags_oris = [self._gradients(img[ch]) for ch in range(c)]
        descs = []
        for y0 in ys:
            for x0 in xs:
                parts = [self._patch_descriptor(mag, ori, y0, x0) for (mag, ori) in mags_oris]
                descs.append(np.concatenate(parts, axis=0))

        return np.concatenate(descs, axis=0).astype(np.float32, copy=False)

    def _gradients(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Central differences with edge padding.
        I = np.asarray(gray, dtype=np.float32)
        gx = np.zeros_like(I)
        gy = np.zeros_like(I)

        gx[:, 1:-1] = I[:, 2:] - I[:, :-2]
        gy[1:-1, :] = I[2:, :] - I[:-2, :]

        mag = np.sqrt(gx * gx + gy * gy)
        ori = np.arctan2(gy, gx)  # [-pi, pi]
        return mag, ori

    def _patch_descriptor(self, mag: np.ndarray, ori: np.ndarray, y0: int, x0: int) -> np.ndarray:
        ps = self.patch_size
        cs = self._cell_size

        m = mag[y0 : y0 + ps, x0 : x0 + ps]
        o = ori[y0 : y0 + ps, x0 : x0 + ps]

        bins = np.linspace(-np.pi, np.pi, self.num_angles + 1, dtype=np.float32)
        parts = []

        for cy in range(self.num_cells):
            for cx in range(self.num_cells):
                yy = slice(cy * cs, (cy + 1) * cs)
                xx = slice(cx * cs, (cx + 1) * cs)
                hist, _ = np.histogram(o[yy, xx], bins=bins, weights=m[yy, xx])
                parts.append(hist.astype(np.float32, copy=False))

        desc = np.concatenate(parts, axis=0)

        # SIFT-style normalisation + clipping.

        nrm = float(np.sqrt((desc * desc).sum()) + self.eps)
        desc = desc / nrm
        if self.clip > 0:
            desc = np.minimum(desc, self.clip)
            desc = desc / float(np.sqrt((desc * desc).sum()) + self.eps)

        return desc


class ClassicSIFT:
    """Classic SIFT-style keypoints + descriptors (NumPy-only).

    Implements a lightweight version of Lowe's SIFT:
      1) Gaussian scale-space + DoG extrema keypoints.
      2) Contrast + edge-response filtering.
      3) Orientation assignment per keypoint.
      4) 4x4x8 = 128-dim descriptor with trilinear (x,y,theta) interpolation,
         L2 normalisation + clipping (L2-Hys).

    Output is fixed-length for kernel pipelines by concatenating the top-K
    descriptors (by |DoG response|) and zero-padding if fewer are found.

    Notes:
        - Runs on grayscale internally (RGB is converted to gray).
        - Designed for small images (e.g., 32x32); default settings are tuned
          for that regime.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        colour_mode: str = "rgb",
        n_keypoints: int = 64,
        n_octaves: int = 3,
        n_scales: int = 3,
        sigma0: float = 1.0,
        contrast_threshold: float = 0.005,
        edge_threshold: float = 10.0,
        base_window: int = 12,
        descriptor_width: int = 4,
        descriptor_bins: int = 8,
        orientation_bins: int = 36,
        orientation_peak_ratio: float = 0.8,
        clip: float = 0.2,
        eps: float = 1e-12,
    ):
        self.image_shape = tuple(int(x) for x in image_shape)
        self.colour_mode = str(colour_mode)
        self.n_keypoints = int(n_keypoints)
        self.n_octaves = int(n_octaves)
        self.n_scales = int(n_scales)
        self.sigma0 = float(sigma0)
        self.contrast_threshold = float(contrast_threshold)
        self.edge_threshold = float(edge_threshold)
        self.base_window = int(base_window)
        self.descriptor_width = int(descriptor_width)
        self.descriptor_bins = int(descriptor_bins)
        self.orientation_bins = int(orientation_bins)
        self.orientation_peak_ratio = float(orientation_peak_ratio)
        self.clip = float(clip)
        self.eps = float(eps)

        c, h, w = self.image_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError("image_shape must be positive")
        if self.colour_mode not in {"rgb", "gray"}:
            raise ValueError("colour_mode must be 'rgb' or 'gray'")
        if self.n_keypoints <= 0:
            raise ValueError("n_keypoints must be positive")
        if self.n_octaves <= 0:
            raise ValueError("n_octaves must be positive")
        if self.n_scales <= 1:
            raise ValueError("n_scales must be >= 2")
        if self.sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        if self.contrast_threshold < 0:
            raise ValueError("contrast_threshold must be non-negative")
        if self.edge_threshold <= 0:
            raise ValueError("edge_threshold must be positive")
        if self.base_window <= 0:
            raise ValueError("base_window must be positive")
        if self.descriptor_width <= 0:
            raise ValueError("descriptor_width must be positive")
        if self.descriptor_bins <= 1:
            raise ValueError("descriptor_bins must be >= 2")
        if self.orientation_bins <= 1:
            raise ValueError("orientation_bins must be >= 2")
        if not (0.0 < self.orientation_peak_ratio <= 1.0):
            raise ValueError("orientation_peak_ratio must be in (0, 1]")

        self._k = 2.0 ** (1.0 / float(self.n_scales))

    def fit(self, X: np.ndarray) -> "ClassicSIFT":
        """No-op fit for transformer-style compatibility."""

        _ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute classic SIFT features for a batch.

        Args:
            X: Array of shape (n, C*H*W) or a single sample (C*H*W,).

        Returns:
            Feature matrix of shape (n, K*128) (or (K*128,) for single input).
        """

        X_arr = np.asarray(X)
        single = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            single = True
        if X_arr.ndim != 2:
            raise ValueError("X must be 1D or 2D")

        c, h, w = self.image_shape
        expected = c * h * w
        if X_arr.shape[1] != expected:
            raise ValueError(f"Expected vectors of length {expected}, got {X_arr.shape[1]}")

        feats = [self._transform_one(X_arr[i]) for i in range(X_arr.shape[0])]
        F = np.stack(feats, axis=0)
        return F[0] if single else F

    def _transform_one(self, v: np.ndarray) -> np.ndarray:
        c, h, w = self.image_shape
        img = np.asarray(v, dtype=np.float32).reshape(c, h, w)
        gray = img.mean(axis=0) if c > 1 else img[0]

        gauss_pyr = self._build_gaussian_pyramid(gray)
        dog_pyr = self._build_dog_pyramid(gauss_pyr)

        keypoints = self._detect_keypoints(dog_pyr)
        if len(keypoints) == 0:
            # Fallback: compute at least one descriptor at the center.
            y0 = int(gray.shape[0] // 2)
            x0 = int(gray.shape[1] // 2)
            keypoints = [
                {
                    "octave": 0,
                    "scale": 1,
                    "y": y0,
                    "x": x0,
                    "sigma": float(self.sigma0),
                    "response": 0.0,
                }
            ]

        # Orientation assignment can yield multiple oriented keypoints.
        oriented = []
        for kp in keypoints:
            oriented.extend(self._assign_orientations(gauss_pyr, kp))

        if len(oriented) == 0:
            # Fallback: keep the best keypoint but force angle=0.
            kp0 = dict(keypoints[0])
            kp0["angle"] = 0.0
            oriented = [kp0]

        # Sort by response and keep top-K.
        oriented.sort(key=lambda d: abs(float(d["response"])), reverse=True)
        oriented = oriented[: self.n_keypoints]

        descs = []
        for okp in oriented:
            desc = self._compute_descriptor(gauss_pyr, okp)
            if desc is not None:
                descs.append(desc)

        out = np.zeros((self.n_keypoints, 128), dtype=np.float32)
        if len(descs) > 0:
            take = min(self.n_keypoints, len(descs))
            out[:take] = np.stack(descs[:take], axis=0)

        return out.reshape(-1)

    def _build_gaussian_pyramid(self, base: np.ndarray) -> list[list[np.ndarray]]:
        """Return pyramid[octave][scale] with (S+3) scales per octave."""

        base_img = np.asarray(base, dtype=np.float32)
        pyr: list[list[np.ndarray]] = []

        # Pre-smooth base image to sigma0.
        img = self._gaussian_blur(base_img, self.sigma0)

        for o in range(self.n_octaves):
            octave_imgs: list[np.ndarray] = [img]

            # Build S+2 additional blurred images (total S+3).
            sig_prev = self.sigma0
            for s in range(1, self.n_scales + 3):
                sig_total = self.sigma0 * (self._k**s)
                sig_inc = float(np.sqrt(max(sig_total * sig_total - sig_prev * sig_prev, 1e-12)))
                img = self._gaussian_blur(img, sig_inc)
                octave_imgs.append(img)
                sig_prev = sig_total

            pyr.append(octave_imgs)

            # Next octave base: downsample from scale s = n_scales.
            next_base = octave_imgs[self.n_scales]
            if min(next_base.shape) < 8:
                break
            img = next_base[::2, ::2]

        return pyr

    @staticmethod
    def _build_dog_pyramid(gauss_pyr: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
        dog: list[list[np.ndarray]] = []
        for octave_imgs in gauss_pyr:
            dogs = [octave_imgs[i + 1] - octave_imgs[i] for i in range(len(octave_imgs) - 1)]
            dog.append(dogs)
        return dog

    def _detect_keypoints(self, dog_pyr: list[list[np.ndarray]]) -> list[dict]:
        """Detect DoG extrema and apply contrast/edge filtering."""

        keypoints: list[dict] = []
        # In classic implementations, the effective DoG contrast threshold is
        # scaled by the number of scales per octave.
        ct = self.contrast_threshold / float(self.n_scales)
        edge_r = self.edge_threshold
        edge_ratio = ((edge_r + 1.0) ** 2) / edge_r

        for o, dogs in enumerate(dog_pyr):
            # dogs has length (S+2), keypoints search in s=1..S.
            for s in range(1, len(dogs) - 1):
                Dm = dogs[s - 1]
                D0 = dogs[s]
                Dp = dogs[s + 1]

                h, w = D0.shape
                if h < 5 or w < 5:
                    continue

                # Avoid borders for finite differences.
                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        val = float(D0[y, x])
                        if abs(val) < ct:
                            continue

                        patch = np.array(
                            [
                                Dm[y - 1 : y + 2, x - 1 : x + 2],
                                D0[y - 1 : y + 2, x - 1 : x + 2],
                                Dp[y - 1 : y + 2, x - 1 : x + 2],
                            ]
                        )

                        if val > 0:
                            if val < float(patch.max()):
                                continue
                        else:
                            if val > float(patch.min()):
                                continue

                        # Edge response elimination via Hessian on D0.
                        dxx = float(D0[y, x + 1] + D0[y, x - 1] - 2.0 * D0[y, x])
                        dyy = float(D0[y + 1, x] + D0[y - 1, x] - 2.0 * D0[y, x])
                        dxy = float((D0[y + 1, x + 1] - D0[y + 1, x - 1] - D0[y - 1, x + 1] + D0[y - 1, x - 1]) * 0.25)
                        tr = dxx + dyy
                        det = dxx * dyy - dxy * dxy
                        if det <= 1e-12:
                            continue
                        if (tr * tr) / det >= edge_ratio:
                            continue

                        sigma = self.sigma0 * (2.0**o) * (self._k**s)
                        keypoints.append(
                            {
                                "octave": int(o),
                                "scale": int(s),
                                "y": int(y),
                                "x": int(x),
                                "sigma": float(sigma),
                                "response": float(val),
                            }
                        )

        # Keep only top candidates before orientation/descriptor.
        keypoints.sort(key=lambda d: abs(float(d["response"])), reverse=True)
        # Allow more here because orientation assignment can split.
        return keypoints[: max(self.n_keypoints * 5, self.n_keypoints)]

    def _assign_orientations(self, gauss_pyr: list[list[np.ndarray]], kp: dict) -> list[dict]:
        """Assign one or more dominant orientations to a keypoint."""

        o = int(kp["octave"])
        s = int(kp["scale"])
        y0 = int(kp["y"])
        x0 = int(kp["x"])
        sigma = float(kp["sigma"])

        img = gauss_pyr[o][s]
        mag, ori = self._grad_mag_ori(img)

        # Orientation window.
        sig = 1.5 * sigma
        rad = int(round(3.0 * sig))
        if rad < 1:
            rad = 1

        h, w = img.shape
        y_start = max(1, y0 - rad)
        y_end = min(h - 2, y0 + rad)
        x_start = max(1, x0 - rad)
        x_end = min(w - 2, x0 + rad)
        if y_start > y_end or x_start > x_end:
            return []

        nb = self.orientation_bins
        hist = np.zeros((nb,), dtype=np.float32)
        two_pi = 2.0 * np.pi
        bin_width = two_pi / float(nb)

        for yy in range(y_start, y_end + 1):
            dy = float(yy - y0)
            for xx in range(x_start, x_end + 1):
                dx = float(xx - x0)
                r2 = dx * dx + dy * dy
                if r2 > float(rad * rad):
                    continue
                wgt = float(np.exp(-r2 / (2.0 * sig * sig)))
                m = float(mag[yy, xx])
                a = float(ori[yy, xx]) % two_pi

                pos = a / bin_width
                b0 = int(np.floor(pos)) % nb
                frac = float(pos - np.floor(pos))
                b1 = (b0 + 1) % nb
                hist[b0] += (1.0 - frac) * wgt * m
                hist[b1] += frac * wgt * m

        mmax = float(hist.max())
        if mmax <= 0:
            out = dict(kp)
            out["angle"] = 0.0
            return [out]

        out_kps: list[dict] = []
        thr = self.orientation_peak_ratio * mmax
        for i in range(nb):
            prev = float(hist[(i - 1) % nb])
            cur = float(hist[i])
            nxt = float(hist[(i + 1) % nb])
            if cur < thr:
                continue
            if not (cur >= prev and cur >= nxt):
                continue

            # Parabolic interpolation around the peak.
            denom = (prev - 2.0 * cur + nxt)
            if abs(denom) < 1e-12:
                off = 0.0
            else:
                off = 0.5 * (prev - nxt) / denom
                off = float(np.clip(off, -0.5, 0.5))

            angle = ((float(i) + off) * bin_width) % two_pi
            out = dict(kp)
            out["angle"] = float(angle)
            out_kps.append(out)

        return out_kps

    def _compute_descriptor(self, gauss_pyr: list[list[np.ndarray]], kp: dict) -> np.ndarray | None:
        """Compute 128-dim SIFT descriptor at an oriented keypoint."""

        o = int(kp["octave"])
        s = int(kp["scale"])
        y0 = int(kp["y"])
        x0 = int(kp["x"])
        sigma = float(kp["sigma"])
        ang0 = float(kp.get("angle", 0.0))

        img = gauss_pyr[o][s]
        mag, ori = self._grad_mag_ori(img)

        d = self.descriptor_width
        nori = self.descriptor_bins

        # For small images, use a base window at sigma0 and scale by sigma/sigma0.
        # This keeps the descriptor reasonably sized on 32x32 inputs.
        scale = float(sigma) / float(self.sigma0)
        win = int(round(float(self.base_window) * scale))
        win = int(np.clip(win, 8, min(img.shape[0], img.shape[1]) - 2))
        rad = win // 2

        h, w = img.shape
        y_start = max(1, y0 - rad)
        y_end = min(h - 2, y0 + rad - 1)
        x_start = max(1, x0 - rad)
        x_end = min(w - 2, x0 + rad - 1)
        if y_start > y_end or x_start > x_end:
            return None

        hist = np.zeros((d, d, nori), dtype=np.float32)

        cos_t = float(np.cos(ang0))
        sin_t = float(np.sin(ang0))
        two_pi = 2.0 * np.pi
        bin_width = two_pi / float(nori)

        # Gaussian weighting within descriptor window.
        sig = 0.5 * float(win)

        # Scale from pixels to descriptor-bin coordinates.
        bin_scale = float(d) / float(win)

        for yy in range(y_start, y_end + 1):
            dy = float(yy - y0)
            for xx in range(x_start, x_end + 1):
                dx = float(xx - x0)

                # Rotate into keypoint frame.
                xr = (cos_t * dx + sin_t * dy)
                yr = (-sin_t * dx + cos_t * dy)

                # Spatial bin coordinates.
                xb = xr * bin_scale + (d / 2.0) - 0.5
                yb = yr * bin_scale + (d / 2.0) - 0.5
                if xb < -1.0 or xb > d or yb < -1.0 or yb > d:
                    continue

                m = float(mag[yy, xx])
                if m <= 0:
                    continue

                # Relative orientation.
                a = (float(ori[yy, xx]) - ang0) % two_pi
                ob = a / bin_width

                ix = int(np.floor(xb))
                iy = int(np.floor(yb))
                io = int(np.floor(ob))
                fx = float(xb - ix)
                fy = float(yb - iy)
                fo = float(ob - io)

                # Gaussian weight.
                wgt = float(np.exp(-(xr * xr + yr * yr) / (2.0 * sig * sig)))
                v = wgt * m

                for dyb in (0, 1):
                    yyb = iy + dyb
                    if yyb < 0 or yyb >= d:
                        continue
                    wy = (1.0 - fy) if dyb == 0 else fy
                    for dxb in (0, 1):
                        xxb = ix + dxb
                        if xxb < 0 or xxb >= d:
                            continue
                        wx = (1.0 - fx) if dxb == 0 else fx
                        for dob in (0, 1):
                            oob = (io + dob) % nori
                            wo = (1.0 - fo) if dob == 0 else fo
                            hist[yyb, xxb, oob] += float(v * wx * wy * wo)

        desc = hist.reshape(-1)
        nrm = float(np.sqrt((desc * desc).sum()) + self.eps)
        desc = desc / nrm
        if self.clip > 0:
            desc = np.minimum(desc, self.clip)
            desc = desc / float(np.sqrt((desc * desc).sum()) + self.eps)
        return desc.astype(np.float32, copy=False)

    @staticmethod
    def _grad_mag_ori(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        I = np.asarray(gray, dtype=np.float32)
        gx = np.zeros_like(I)
        gy = np.zeros_like(I)
        gx[:, 1:-1] = I[:, 2:] - I[:, :-2]
        gy[1:-1, :] = I[2:, :] - I[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        ori = np.arctan2(gy, gx)
        return mag, ori

    @staticmethod
    def _gaussian_kernel1d(sigma: float) -> np.ndarray:
        sig = float(sigma)
        if sig <= 0:
            raise ValueError("sigma must be positive")
        radius = int(np.ceil(3.0 * sig))
        if radius < 1:
            radius = 1
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-(x * x) / (2.0 * sig * sig)).astype(np.float32)
        k = k / float(k.sum())
        return k

    @classmethod
    def _gaussian_blur(cls, img: np.ndarray, sigma: float) -> np.ndarray:
        """Separable Gaussian blur with reflect padding (NumPy-only)."""

        I = np.asarray(img, dtype=np.float32)
        k = cls._gaussian_kernel1d(float(sigma))
        r = int(k.shape[0] // 2)

        # Convolve rows.
        padded = np.pad(I, ((0, 0), (r, r)), mode="reflect")
        tmp = np.empty_like(I)
        for i in range(I.shape[0]):
            tmp[i] = np.convolve(padded[i], k, mode="valid")

        # Convolve cols.
        padded = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
        out = np.empty_like(I)
        for j in range(I.shape[1]):
            out[:, j] = np.convolve(padded[:, j], k, mode="valid")

        return out


class HOG:
    """Histogram of Oriented Gradients (HOG) descriptor on a fixed grid.

    Designed for small images represented as flattened vectors in CHW order
    (e.g., 3x32x32). The descriptor:
      1) Computes per-pixel gradients.
      2) Accumulates orientation histograms over cells.
      3) Concatenates block-normalised cell histograms (L2-Hys).

    Notes:
      - This is a classic HOG variant; it uses simple spatial assignment (no
        bilinear spatial interpolation) and linear interpolation between
        orientation bins.
      - Output features are non-negative.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        colour_mode: str = "rgb",
        cell_size: int = 8,
        block_size: int = 2,
        block_stride: int = 1,
        num_bins: int = 9,
        signed: bool = False,
        clip: float = 0.2,
        eps: float = 1e-12,
    ):
        self.image_shape = tuple(int(x) for x in image_shape)
        self.colour_mode = str(colour_mode)
        self.cell_size = int(cell_size)
        self.block_size = int(block_size)
        self.block_stride = int(block_stride)
        self.num_bins = int(num_bins)
        self.signed = bool(signed)
        self.clip = float(clip)
        self.eps = float(eps)

        c, h, w = self.image_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError("image_shape must be positive")
        if self.colour_mode not in {"rgb", "gray"}:
            raise ValueError("colour_mode must be 'rgb' or 'gray'")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if h % self.cell_size != 0 or w % self.cell_size != 0:
            raise ValueError("image dimensions must be divisible by cell_size")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.block_stride <= 0:
            raise ValueError("block_stride must be positive")
        if self.num_bins <= 1:
            raise ValueError("num_bins must be >= 2")

        self._n_cells_y = h // self.cell_size
        self._n_cells_x = w // self.cell_size
        if self._n_cells_y < self.block_size or self._n_cells_x < self.block_size:
            raise ValueError("block_size too large for the number of cells")

    def fit(self, X: np.ndarray) -> "HOG":
        """No-op fit for transformer-style compatibility."""

        _ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute HOG features for a batch.

        Args:
            X: Array of shape (n, C*H*W) or a single sample (C*H*W,).

        Returns:
            Feature matrix of shape (n, d_out) (or (d_out,) for single input).
        """

        X_arr = np.asarray(X)
        single = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            single = True
        if X_arr.ndim != 2:
            raise ValueError("X must be 1D or 2D")

        c, h, w = self.image_shape
        expected = c * h * w
        if X_arr.shape[1] != expected:
            raise ValueError(f"Expected vectors of length {expected}, got {X_arr.shape[1]}")

        feats = [self._transform_one(X_arr[i]) for i in range(X_arr.shape[0])]
        F = np.stack(feats, axis=0)
        return F[0] if single else F

    def _transform_one(self, v: np.ndarray) -> np.ndarray:
        c, h, w = self.image_shape
        img = np.asarray(v, dtype=np.float32).reshape(c, h, w)

        mag, ori = self._gradients_orientation(img)

        # Cell histograms: (cy, cx, nb)
        Hc = self._cell_histograms(mag, ori)

        # Block normalisation (L2-Hys) and concatenation.
        bs = self.block_size
        st = self.block_stride
        nby = (self._n_cells_y - bs) // st + 1
        nbx = (self._n_cells_x - bs) // st + 1
        block_dim = bs * bs * self.num_bins

        out = np.empty((nby * nbx, block_dim), dtype=np.float32)
        t = 0
        for by in range(nby):
            y0 = by * st
            for bx in range(nbx):
                x0 = bx * st
                block = Hc[y0 : y0 + bs, x0 : x0 + bs].reshape(-1)

                nrm = float(np.sqrt((block * block).sum()) + self.eps)
                block = block / nrm
                if self.clip > 0:
                    block = np.minimum(block, self.clip)
                    block = block / float(np.sqrt((block * block).sum()) + self.eps)

                out[t] = block
                t += 1

        return out.reshape(-1)

    def _gradients_orientation(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradient magnitude and orientation in radians."""

        c, _, _ = self.image_shape

        if c == 1 or self.colour_mode == "gray":
            gray = img.mean(axis=0) if img.shape[0] > 1 else img[0]
            gx, gy = self._gradients(gray)
            mag = np.sqrt(gx * gx + gy * gy)
            ori = np.arctan2(gy, gx)
            return mag, ori

        # RGB: compute per-channel gradients, take the channel with maximum magnitude.
        mags = []
        oris = []
        for ch in range(img.shape[0]):
            gx, gy = self._gradients(img[ch])
            m = np.sqrt(gx * gx + gy * gy)
            o = np.arctan2(gy, gx)
            mags.append(m)
            oris.append(o)

        mag_stack = np.stack(mags, axis=0)  # (C, H, W)
        ori_stack = np.stack(oris, axis=0)  # (C, H, W)
        arg = np.argmax(mag_stack, axis=0)  # (H, W)

        mag = np.take_along_axis(mag_stack, arg[None, :, :], axis=0)[0]
        ori = np.take_along_axis(ori_stack, arg[None, :, :], axis=0)[0]
        return mag, ori

    @staticmethod
    def _gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Central differences with edge padding (same as DenseSIFT)."""

        I = np.asarray(gray, dtype=np.float32)
        gx = np.zeros_like(I)
        gy = np.zeros_like(I)

        gx[:, 1:-1] = I[:, 2:] - I[:, :-2]
        gy[1:-1, :] = I[2:, :] - I[:-2, :]
        return gx, gy

    def _cell_histograms(self, mag: np.ndarray, ori: np.ndarray) -> np.ndarray:
        """Accumulate orientation histograms per cell."""

        h = mag.shape[0]
        w = mag.shape[1]
        cs = self.cell_size
        nb = self.num_bins

        # Map orientations into [0, range).
        if self.signed:
            ang_range = 2.0 * np.pi
            ang = (ori + 2.0 * np.pi) % (2.0 * np.pi)
        else:
            ang_range = np.pi
            ang = ori % np.pi

        bin_width = ang_range / float(nb)
        pos = ang / bin_width
        b0 = np.floor(pos).astype(np.int32) % nb
        frac = (pos - np.floor(pos)).astype(np.float32)
        b1 = (b0 + 1) % nb

        w0 = (1.0 - frac) * mag
        w1 = frac * mag

        cy = (np.arange(h)[:, None] // cs).astype(np.int32)
        cx = (np.arange(w)[None, :] // cs).astype(np.int32)
        cy = np.broadcast_to(cy, (h, w))
        cx = np.broadcast_to(cx, (h, w))

        Hc = np.zeros((self._n_cells_y, self._n_cells_x, nb), dtype=np.float32)

        # Flatten and accumulate with np.add.at.
        idx_cy = cy.reshape(-1)
        idx_cx = cx.reshape(-1)
        idx_b0 = b0.reshape(-1)
        idx_b1 = b1.reshape(-1)
        val0 = w0.reshape(-1)
        val1 = w1.reshape(-1)

        np.add.at(Hc, (idx_cy, idx_cx, idx_b0), val0)
        np.add.at(Hc, (idx_cy, idx_cx, idx_b1), val1)

        return Hc


class HSVHistogram:
    """HSV colour histogram descriptor (NumPy-only).

    Designed for small images represented as flattened vectors in CHW order
    (e.g., 3x32x32). The descriptor computes an HSV histogram either globally
    or over a spatial grid (e.g., 2x2) and concatenates region histograms.

    Defaults are chosen to pair well with histogram kernels (e.g., chi2-RBF):
      - Non-negative features.
      - L1 normalisation per region.
      - Automatic input scaling to [0, 1].

    Args:
        image_shape: (C, H, W), typically (3, 32, 32).
        colour_mode: 'rgb' or 'gray'. If 'gray' (or C==1), uses V=gray and
            sets H=S=0 (so only V bins vary).
        bins: (h_bins, s_bins, v_bins). Default (8, 4, 4).
        grid: (gy, gx) spatial grid. (1, 1) is global; (2, 2) is 4 regions.
        eps: Small constant for numerical stability.

    Example:
        hsv = HSVHistogram(bins=(8, 8, 8), grid=(2, 2)).fit(X_tr)
        X_tr_hsv = hsv.transform(X_tr)
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        colour_mode: str = "rgb",
        bins: tuple[int, int, int] = (8, 4, 4),
        grid: tuple[int, int] = (1, 1),
        eps: float = 1e-12,
    ):
        self.image_shape = tuple(int(x) for x in image_shape)
        self.colour_mode = str(colour_mode)
        self.bins = (int(bins[0]), int(bins[1]), int(bins[2]))
        self.grid = (int(grid[0]), int(grid[1]))
        self.eps = float(eps)

        c, h, w = self.image_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError("image_shape must be positive")
        if self.colour_mode not in {"rgb", "gray"}:
            raise ValueError("colour_mode must be 'rgb' or 'gray'")
        hb, sb, vb = self.bins
        if hb <= 0 or sb <= 0 or vb <= 0:
            raise ValueError("bins must be positive")
        gy, gx = self.grid
        if gy <= 0 or gx <= 0:
            raise ValueError("grid must be positive")
        if h % gy != 0 or w % gx != 0:
            raise ValueError("grid must evenly divide image H and W")

        self._region_h = h // gy
        self._region_w = w // gx
        self._region_dim = hb * sb * vb

    def fit(self, X: np.ndarray) -> "HSVHistogram":
        """No-op fit for transformer-style compatibility."""

        _ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute HSV histogram features for a batch.

        Args:
            X: Array of shape (n, C*H*W) or a single sample (C*H*W,).

        Returns:
            Feature matrix of shape (n, d_out) (or (d_out,) for single input).
        """

        X_arr = np.asarray(X)
        single = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            single = True
        if X_arr.ndim != 2:
            raise ValueError("X must be 1D or 2D")

        c, h, w = self.image_shape
        expected = c * h * w
        if X_arr.shape[1] != expected:
            raise ValueError(f"Expected vectors of length {expected}, got {X_arr.shape[1]}")

        feats = [self._transform_one(X_arr[i]) for i in range(X_arr.shape[0])]
        F = np.stack(feats, axis=0)
        return F[0] if single else F

    def _transform_one(self, v: np.ndarray) -> np.ndarray:
        c, h, w = self.image_shape
        img = np.asarray(v, dtype=np.float32).reshape(c, h, w)

        if c == 1 or self.colour_mode == "gray":
            gray = img.mean(axis=0) if c > 1 else img[0]
            gray = self._to_unit_float(gray)
            hh = np.zeros_like(gray)
            ss = np.zeros_like(gray)
            vv = gray
        else:
            if c < 3:
                raise ValueError("RGB mode requires at least 3 channels")
            rgb = img[:3]
            rgb = self._to_unit_float(rgb)
            rgb = np.transpose(rgb, (1, 2, 0))  # HWC
            hh, ss, vv = self._rgb_to_hsv(rgb)

        hb, sb, vb = self.bins
        gy, gx = self.grid
        region_h = self._region_h
        region_w = self._region_w

        out = np.empty((gy * gx, self._region_dim), dtype=np.float32)
        t = 0
        for ry in range(gy):
            y0 = ry * region_h
            y1 = y0 + region_h
            for rx in range(gx):
                x0 = rx * region_w
                x1 = x0 + region_w
                hist = self._hist3d(hh[y0:y1, x0:x1], ss[y0:y1, x0:x1], vv[y0:y1, x0:x1], hb, sb, vb)
                out[t] = hist
                t += 1

        return out.reshape(-1)

    def _hist3d(
        self,
        h: np.ndarray,
        s: np.ndarray,
        v: np.ndarray,
        hb: int,
        sb: int,
        vb: int,
    ) -> np.ndarray:
        # Quantise into {0, ..., bins-1}. Important: values can be exactly 1.0
        # after float32 rounding, so we clamp to the last bin explicitly.
        hq = np.minimum((np.clip(h, 0.0, 1.0) * float(hb)).astype(np.int32), hb - 1)
        sq = np.minimum((np.clip(s, 0.0, 1.0) * float(sb)).astype(np.int32), sb - 1)
        vq = np.minimum((np.clip(v, 0.0, 1.0) * float(vb)).astype(np.int32), vb - 1)

        idx = (hq * (sb * vb) + sq * vb + vq).reshape(-1)
        hist = np.bincount(idx, minlength=hb * sb * vb).astype(np.float32, copy=False)

        # L1 normalisation (sum=1) per region.
        hist = hist / float(hist.sum() + self.eps)
        return hist

    @staticmethod
    def _to_unit_float(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        mx = float(np.max(x))
        if mx > 1.0 + 1e-6:
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)

    def _rgb_to_hsv(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        diff = mx - mn

        v = mx
        s = np.where(mx > self.eps, diff / (mx + self.eps), 0.0).astype(np.float32)

        h = np.zeros_like(mx, dtype=np.float32)
        mask = diff > self.eps
        # Avoid division by zero by only computing where mask is true.
        diff_safe = np.where(mask, diff, 1.0)

        rmax = (mx == r) & mask
        gmax = (mx == g) & mask
        bmax = (mx == b) & mask

        h_r = ((g - b) / diff_safe) % 6.0
        h_g = ((b - r) / diff_safe) + 2.0
        h_b = ((r - g) / diff_safe) + 4.0

        h = np.where(rmax, h_r, h)
        h = np.where(gmax, h_g, h)
        h = np.where(bmax, h_b, h)

        h = (h / 6.0) % 1.0
        return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


class UniformLBPHistogram:
    """Uniform LBP (u2) histogram descriptor (NumPy-only).

    Computes Local Binary Pattern (LBP) codes on a per-pixel basis using P neighbours
    on a circle of radius R (this implementation currently uses the standard 8-neighbour
    layout for R=1). Codes are mapped to the "uniform" (u2) set: patterns with <=2
    bit transitions are assigned individual bins, while all non-uniform patterns go
    into a final catch-all bin.

    The histogram can be computed per spatial cell and either:
      - pooled by summation over cells (pool="sum"), producing a single histogram, or
      - concatenated across cells (pool="concat"), preserving coarse spatial layout.

    By default the descriptor is designed for histogram kernels (e.g., chi2-RBF):
      - non-negative features
      - L1 normalisation

    Args:
        image_shape: (C, H, W), typically (3, 32, 32).
        colour_mode: 'rgb' or 'gray'. If 'gray' (or C==1), uses a single channel.
        per_channel: If True and colour_mode='rgb', compute LBP per channel and
            concatenate channel histograms.
        cell_size: Cell size in pixels, either int or (cell_h, cell_w). Must evenly
            divide H and W.
        pool: "sum" or "concat".
        P: Number of neighbours (currently only P=8 is supported).
        R: Radius (currently only R=1 is supported).
        eps: Small constant for numerical stability.

    Output dimension:
                Let B be the number of bins after uniform mapping (for P=8, this
                implementation yields B=59: 58 uniform + 1 non-uniform). Let C' be 1 if not
        per_channel else 3. Let Ny = H/cell_h and Nx = W/cell_w.
          - pool="sum":    d_out = B * C'
          - pool="concat": d_out = (Ny*Nx) * B * C'
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        colour_mode: str = "rgb",
        per_channel: bool = False,
        cell_size: int | tuple[int, int] = 8,
        pool: str = "sum",
        P: int = 8,
        R: int = 1,
        eps: float = 1e-12,
    ):
        self.image_shape = tuple(int(x) for x in image_shape)
        self.colour_mode = str(colour_mode)
        self.per_channel = bool(per_channel)
        self.pool = str(pool)
        self.P = int(P)
        self.R = int(R)
        self.eps = float(eps)

        if isinstance(cell_size, tuple):
            self.cell_size = (int(cell_size[0]), int(cell_size[1]))
        else:
            self.cell_size = (int(cell_size), int(cell_size))

        c, h, w = self.image_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError("image_shape must be positive")
        if self.colour_mode not in {"rgb", "gray"}:
            raise ValueError("colour_mode must be 'rgb' or 'gray'")
        if self.pool not in {"sum", "concat"}:
            raise ValueError("pool must be 'sum' or 'concat'")

        ch, cw = self.cell_size
        if ch <= 0 or cw <= 0:
            raise ValueError("cell_size must be positive")
        if h % ch != 0 or w % cw != 0:
            raise ValueError("cell_size must evenly divide image H and W")

        if self.P != 8 or self.R != 1:
            raise ValueError("This implementation currently supports only P=8 and R=1")

        self._n_cells_y = h // ch
        self._n_cells_x = w // cw

        self._mapping, self._n_bins = self._make_uniform_mapping(P=self.P)

    def fit(self, X: np.ndarray) -> "UniformLBPHistogram":
        _ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        single = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            single = True
        if X_arr.ndim != 2:
            raise ValueError("X must be 1D or 2D")

        c, h, w = self.image_shape
        expected = c * h * w
        if X_arr.shape[1] != expected:
            raise ValueError(f"Expected vectors of length {expected}, got {X_arr.shape[1]}")

        feats = [self._transform_one(X_arr[i]) for i in range(X_arr.shape[0])]
        F = np.stack(feats, axis=0)
        return F[0] if single else F

    def _transform_one(self, v: np.ndarray) -> np.ndarray:
        c, h, w = self.image_shape
        img = np.asarray(v, dtype=np.float32).reshape(c, h, w)

        # Decide channels to process.
        if c == 1 or self.colour_mode == "gray":
            channels = [self._to_gray(img)]
        else:
            if c < 3:
                raise ValueError("RGB mode requires at least 3 channels")
            if self.per_channel:
                rgb = self._to_unit_float(img[:3])
                channels = [rgb[0], rgb[1], rgb[2]]
            else:
                channels = [self._to_gray(img[:3])]

        parts = [self._lbp_hist_one_channel(ch) for ch in channels]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def _lbp_hist_one_channel(self, gray: np.ndarray) -> np.ndarray:
        # gray is (H,W) float32 in [0,1]
        code = self._lbp_code(gray)
        mapped = self._mapping[code]

        ch, cw = self.cell_size
        ny, nx = self._n_cells_y, self._n_cells_x
        B = int(self._n_bins)

        if self.pool == "sum":
            hist = np.zeros((B,), dtype=np.float32)
            for cy in range(ny):
                y0, y1 = cy * ch, (cy + 1) * ch
                for cx in range(nx):
                    x0, x1 = cx * cw, (cx + 1) * cw
                    hcell = np.bincount(mapped[y0:y1, x0:x1].reshape(-1), minlength=B).astype(
                        np.float32, copy=False
                    )
                    hist += hcell
            hist = hist / float(hist.sum() + self.eps)
            return hist

        # pool == "concat": L1-normalise each cell histogram, then concatenate.
        out = np.empty((ny * nx, B), dtype=np.float32)
        t = 0
        for cy in range(ny):
            y0, y1 = cy * ch, (cy + 1) * ch
            for cx in range(nx):
                x0, x1 = cx * cw, (cx + 1) * cw
                hcell = np.bincount(mapped[y0:y1, x0:x1].reshape(-1), minlength=B).astype(
                    np.float32, copy=False
                )
                hcell = hcell / float(hcell.sum() + self.eps)
                out[t] = hcell
                t += 1
        return out.reshape(-1)

    def _lbp_code(self, gray: np.ndarray) -> np.ndarray:
        # Standard 8-neighbour LBP at radius 1 with edge padding.
        g = self._to_unit_float(gray)
        pad = int(self.R)
        gp = np.pad(g, ((pad, pad), (pad, pad)), mode="edge")

        H, W = g.shape
        c0 = gp[pad : pad + H, pad : pad + W]

        # Neighbour offsets in clockwise order starting at top-left.
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        code = np.zeros((H, W), dtype=np.uint8)
        for bit, (dy, dx) in enumerate(offsets):
            nb = gp[pad + dy : pad + dy + H, pad + dx : pad + dx + W]
            code |= ((nb >= c0).astype(np.uint8) << np.uint8(bit))
        return code

    @staticmethod
    def _make_uniform_mapping(*, P: int) -> tuple[np.ndarray, int]:
        if P <= 0 or P > 16:
            raise ValueError("P must be in [1,16]")
        n_codes = 1 << int(P)

        mapping = np.empty((n_codes,), dtype=np.int32)

        # Find uniform codes (<=2 transitions in circular bit string).
        uniform_codes: list[int] = []
        for code in range(n_codes):
            bits = ((code >> np.arange(P, dtype=np.int32)) & 1).astype(np.int32)
            transitions = int(np.sum(bits != np.roll(bits, 1)))
            if transitions <= 2:
                uniform_codes.append(int(code))

        # All non-uniform patterns map to last bin.
        non_uniform_bin = int(len(uniform_codes))
        mapping.fill(non_uniform_bin)
        for idx, code in enumerate(uniform_codes):
            mapping[int(code)] = int(idx)

        n_bins = non_uniform_bin + 1
        return mapping, n_bins

    @staticmethod
    def _to_unit_float(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        mx = float(np.max(x))
        if mx > 1.0 + 1e-6:
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)

    def _to_gray(self, img_chw: np.ndarray) -> np.ndarray:
        # img_chw: (C,H,W)
        x = self._to_unit_float(img_chw)
        if x.ndim == 2:
            return x
        if x.shape[0] == 1:
            return x[0]
        return x.mean(axis=0)


class ColorMoments:
    """Lab colour moments descriptor (NumPy-only).

    Computes per-channel moments (mean, variance, skewness) in CIE Lab space,
    either globally or over a spatial grid, then concatenates the results.

    This is a classic descriptor in the literature ("color moments"), often
    used as a lightweight complement to gradient-based descriptors like HOG.

    Args:
        image_shape: (C, H, W), typically (3, 32, 32).
        colour_mode: 'rgb' or 'gray'. If 'gray' (or C==1), the image is treated
            as grayscale; Lab is approximated by L=gray, a=b=0.
        grid: (gy, gx) spatial grid. (1, 1) is global; (2, 2) is 4 regions.
        eps: Small constant for numerical stability.

    Notes:
        - Skewness is the normalised third central moment:
                    skew = E[(x - mu)^3] / (sigma^3 + eps)
        - Output dimension is (gy*gx) * (3 channels) * (3 moments) = 9*gy*gx.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        colour_mode: str = "rgb",
        grid: tuple[int, int] = (1, 1),
        eps: float = 1e-12,
    ):
        self.image_shape = tuple(int(x) for x in image_shape)
        self.colour_mode = str(colour_mode)
        self.grid = (int(grid[0]), int(grid[1]))
        self.eps = float(eps)

        c, h, w = self.image_shape
        if c <= 0 or h <= 0 or w <= 0:
            raise ValueError("image_shape must be positive")
        if self.colour_mode not in {"rgb", "gray"}:
            raise ValueError("colour_mode must be 'rgb' or 'gray'")
        gy, gx = self.grid
        if gy <= 0 or gx <= 0:
            raise ValueError("grid must be positive")
        if h % gy != 0 or w % gx != 0:
            raise ValueError("grid must evenly divide image H and W")

        self._region_h = h // gy
        self._region_w = w // gx

    def fit(self, X: np.ndarray) -> "ColorMoments":
        """No-op fit for transformer-style compatibility."""

        _ = X
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute Lab colour moments for a batch.

        Args:
            X: Array of shape (n, C*H*W) or a single sample (C*H*W,).

        Returns:
            Feature matrix of shape (n, d_out) (or (d_out,) for single input).
        """

        X_arr = np.asarray(X)
        single = False
        if X_arr.ndim == 1:
            X_arr = X_arr[None, :]
            single = True
        if X_arr.ndim != 2:
            raise ValueError("X must be 1D or 2D")

        c, h, w = self.image_shape
        expected = c * h * w
        if X_arr.shape[1] != expected:
            raise ValueError(f"Expected vectors of length {expected}, got {X_arr.shape[1]}")

        feats = [self._transform_one(X_arr[i]) for i in range(X_arr.shape[0])]
        F = np.stack(feats, axis=0)
        return F[0] if single else F

    def _transform_one(self, v: np.ndarray) -> np.ndarray:
        c, h, w = self.image_shape
        img = np.asarray(v, dtype=np.float32).reshape(c, h, w)

        if c == 1 or self.colour_mode == "gray":
            gray = img.mean(axis=0) if c > 1 else img[0]
            gray = self._to_unit_float(gray)
            L = gray * 100.0
            a = np.zeros_like(L)
            b = np.zeros_like(L)
        else:
            if c < 3:
                raise ValueError("RGB mode requires at least 3 channels")
            rgb = img[:3]
            rgb = self._to_unit_float(rgb)
            rgb = np.transpose(rgb, (1, 2, 0))  # HWC in [0,1]
            L, a, b = self._rgb_to_lab(rgb)

        gy, gx = self.grid
        rh = self._region_h
        rw = self._region_w

        out = np.empty((gy * gx, 9), dtype=np.float32)
        t = 0
        for ry in range(gy):
            y0 = ry * rh
            y1 = y0 + rh
            for rx in range(gx):
                x0 = rx * rw
                x1 = x0 + rw
                feats = self._moments_region(
                    L[y0:y1, x0:x1],
                    a[y0:y1, x0:x1],
                    b[y0:y1, x0:x1],
                )
                out[t] = feats
                t += 1

        return out.reshape(-1)

    def _moments_region(self, c0: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        m0 = self._moments_1d(c0.reshape(-1))
        m1 = self._moments_1d(c1.reshape(-1))
        m2 = self._moments_1d(c2.reshape(-1))
        return np.concatenate([m0, m1, m2], axis=0).astype(np.float32, copy=False)

    def _moments_1d(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        mu = float(xx.mean())
        centered = xx - mu
        var = float(np.mean(centered * centered))
        sig = float(np.sqrt(var + self.eps))
        skew = float(np.mean(centered * centered * centered) / (sig**3 + self.eps))
        return np.array([mu, var, skew], dtype=np.float32)

    @staticmethod
    def _to_unit_float(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        mx = float(np.max(x))
        if mx > 1.0 + 1e-6:
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)

    def _rgb_to_lab(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert sRGB in [0,1] to CIE Lab (D65)."""

        # sRGB companding
        rgb = np.asarray(rgb, dtype=np.float32)
        a = 0.055
        rgb_lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1.0 + a)) ** 2.4)

        # Linear RGB to XYZ (D65)
        r = rgb_lin[..., 0]
        g = rgb_lin[..., 1]
        b = rgb_lin[..., 2]

        X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        # Normalize by D65 white point
        Xn = 0.95047
        Yn = 1.0
        Zn = 1.08883
        x = X / Xn
        y = Y / Yn
        z = Z / Zn

        delta = 6.0 / 29.0
        delta3 = delta**3
        inv_3delta2 = 1.0 / (3.0 * (delta**2))

        def f(t: np.ndarray) -> np.ndarray:
            return np.where(t > delta3, np.cbrt(t), t * inv_3delta2 + 4.0 / 29.0)

        fx = f(x)
        fy = f(y)
        fz = f(z)

        L = 116.0 * fy - 16.0
        aa = 500.0 * (fx - fy)
        bb = 200.0 * (fy - fz)
        return L.astype(np.float32), aa.astype(np.float32), bb.astype(np.float32)
