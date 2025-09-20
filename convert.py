#!/usr/bin/env python3
"""
DynatopGIS Converter - Production Ready Version with Cycle Fix
Complete implementation with cycle prevention through edge voting
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import rasterio
from scipy import ndimage
from collections import deque, defaultdict


class DynatopGISConverter:
    """Convert HRU delineation to dynatopGIS format with cycle prevention"""

    # Configurable parameters
    WIDTH_CAP_FACTOR = 6.0
    CHANNEL_OVERLAP_MIN = 0.02
    DOMINANT_SPLIT_THRESH = 0.7
    MAX_OUTLETS_BEFORE_CONSOLIDATE = 5
    SINGLE_DOWNSTREAM_MODE = False  # Set to True to enforce single downstream per HRU

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self.rasters = {}
        self.hru_stats = None
        self.hru_network = {}
        self.orig_network = {}
        self.id_remap = {}
        self.id_remap_inv = {}

        # Metadata
        self.crs = None
        self.transform = None
        self.res = None
        self.shape = None

        # Reproducible random generator
        self._rng = np.random.default_rng(12345)

        # 4-connected structure for morphology
        self._S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

        # D8 direction mappings (different conventions)
        self._d8_maps = [
            # Whitebox/TauDEM style (powers of 2)
            {1: (0, 1), 2: (-1, 1), 4: (-1, 0), 8: (-1, -1), 16: (0, -1), 32: (1, -1), 64: (1, 0), 128: (1, 1)},
            # ESRI ArcGIS style (powers of 2)
            {1: (-1, 0), 2: (-1, 1), 4: (0, 1), 8: (1, 1), 16: (1, 0), 32: (1, -1), 64: (0, -1), 128: (-1, -1)}
        ]

        # ADD: ESRI 1-8 style (WhiteboxTools d8_pointer output)
        self._d8_esri_1to8 = {
            1: (0, 1),  # E
            2: (1, 1),  # SE
            3: (1, 0),  # S
            4: (1, -1),  # SW
            5: (0, -1),  # W
            6: (-1, -1),  # NW
            7: (-1, 0),  # N
            8: (-1, 1)  # NE
        }

        self._d8_offsets = None

        # Cache for HRU masks/edges
        self._hru_mask_cache = {}
        self._hru_edge_cache = {}

    def _same_transform(self, t1, t2, eps=1e-7):
        """Check if two transforms are effectively the same"""
        return all(abs(a - b) <= eps for a, b in zip(t1, t2))

    def _get_hru_masks(self, oid):
        """Cache and return HRU mask and edge"""
        if oid not in self._hru_mask_cache:
            m = (self.rasters['hrus'] == oid)
            e = m & ~ndimage.binary_erosion(m, structure=self._S4)
            self._hru_mask_cache[oid] = m
            self._hru_edge_cache[oid] = e
        return self._hru_mask_cache[oid], self._hru_edge_cache[oid]

    def load_data(self):
        """Load all required data with validation and NoData handling"""
        print("\nLoading data...")

        # Load HRU statistics
        stats_file = self.input_dir / 'hru_statistics.csv'
        if not stats_file.exists():
            raise FileNotFoundError(f"HRU statistics not found: {stats_file}")

        self.hru_stats = pd.read_csv(stats_file)
        print(f"  Loaded {len(self.hru_stats)} HRUs")

        # Load rasters with NoData masking
        raster_files = {
            'hrus': 'hrus.tif',
            'twi': 'twi.tif',
            'slope': 'slope.tif',
            'flow_dir': 'flow_direction.tif',
            'flow_acc': 'flow_accumulation.tif',
            'rivers': 'rivers_from_shapefile.tif'
        }

        for key, filename in raster_files.items():
            filepath = self.input_dir / filename
            if not filepath.exists() and key == 'rivers':
                filepath = self.input_dir / 'streams.tif'

            if filepath.exists():
                with rasterio.open(filepath) as src:
                    arr = src.read(1)
                    nodata = src.nodata

                    # Handle NoData values
                    if nodata is not None:
                        arr = np.where(arr == nodata, np.nan, arr)

                    self.rasters[key] = arr

                    if self.crs is None:
                        self.crs = src.crs
                        self.transform = src.transform
                        self.res = (src.transform[0], abs(src.transform[4]))
                        self.shape = src.shape
                    else:
                        # Validate alignment
                        if src.shape != self.shape:
                            raise ValueError(f"Raster {key} shape {src.shape} != base {self.shape}")
                        if not self._same_transform(src.transform, self.transform):
                            raise ValueError(f"Raster {key} has different transform")

                print(f"  Loaded {key}: {filepath.name}")
            else:
                print(f"  Warning: {key} not found")

        # Verify CRS and resolution
        if self.crs and hasattr(self.crs, 'is_geographic') and self.crs.is_geographic:
            raise ValueError("HRU rasters use geographic CRS. Reproject to projected CRS (meters) first.")

        print(f"  Resolution: {self.res[0]:.1f} x {self.res[1]:.1f} meters")
        if self.res[0] < 1 or self.res[0] > 1000:
            print(f"  WARNING: Unusual resolution {self.res[0]}m - verify CRS is in meters")

    def _d8_pointer_votes(self, d8_map, sample_n=500):
        """Test D8 map to determine flow direction votes"""
        if 'flow_acc' not in self.rasters or 'flow_dir' not in self.rasters:
            return 0

        fa = self.rasters['flow_acc']
        fd = self.rasters['flow_dir']
        hrus = self.rasters['hrus']

        rows, cols = np.where((hrus > 0) & np.isfinite(fd))
        if len(rows) == 0:
            return 0

        sample_idx = self._rng.choice(len(rows), size=min(sample_n, len(rows)), replace=False)
        votes = 0

        for i in sample_idx:
            r, c = rows[i], cols[i]
            code = int(fd[r, c])

            if code not in d8_map:
                continue

            dr, dc = d8_map[code]
            nr, nc = r + dr, c + dc

            if 0 <= nr < fd.shape[0] and 0 <= nc < fd.shape[1]:
                if np.isfinite(fa[nr, nc]) and np.isfinite(fa[r, c]):
                    # Tolerance for flat areas
                    if fa[nr, nc] >= fa[r, c] - 1e-9:
                        votes += 1
                    else:
                        votes -= 1

        return votes

    def _choose_d8_offsets(self):
        """Auto-detect best D8 direction mapping from powers-of-2 styles"""
        if 'flow_dir' not in self.rasters or 'flow_acc' not in self.rasters:
            self._d8_offsets = self._d8_maps[0]
            print("  Using default D8 mapping (Whitebox style)")
            return

        best_votes, best_idx = -np.inf, 0
        for idx, d8_map in enumerate(self._d8_maps):
            votes = self._d8_pointer_votes(d8_map)
            if votes > best_votes:
                best_votes, best_idx = votes, idx

        self._d8_offsets = self._d8_maps[best_idx]
        style = "Whitebox" if best_idx == 0 else "ESRI"
        print(f"  Auto-detected D8 style: {style}")

    def _detect_d8_direction(self):
        """Detect if D8 codes point outflow or inflow"""
        if not self._d8_offsets:
            self._choose_d8_offsets()

        votes = self._d8_pointer_votes(self._d8_offsets)
        return votes >= 0

    def _has_cycle(self):
        """Check for cycles in flow network using topological sort"""
        indeg = {u: 0 for u in self.hru_network}
        for u, info in self.hru_network.items():
            for v in info['downstream']:
                indeg[v] += 1

        q = deque([u for u, d in indeg.items() if d == 0])
        seen = 0

        while q:
            u = q.popleft()
            seen += 1
            for v in self.hru_network[u]['downstream']:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        return seen != len(self.hru_network)

    def _sccs(self):
        """Tarjan's algorithm for finding strongly connected components"""
        index = 0
        stack, onstack = [], set()
        idx, low = {}, {}
        comps = []

        def strongconnect(v):
            nonlocal index
            idx[v] = index
            low[v] = index
            index += 1
            stack.append(v)
            onstack.add(v)

            for w in self.hru_network[v]['downstream']:
                if w not in idx:
                    strongconnect(w)
                    low[v] = min(low[v], low[w])
                elif w in onstack:
                    low[v] = min(low[v], idx[w])

            if low[v] == idx[v]:
                comp = set()
                while True:
                    x = stack.pop()
                    onstack.remove(x)
                    comp.add(x)
                    if x == v:
                        break
                comps.append(comp)

        for v in self.hru_network:
            if v not in idx:
                strongconnect(v)
        return comps

    def _force_acyclic(self, edge_counts):
        """Remove minimum-weight edges from SCCs until DAG is achieved"""
        # Upper bound: can't cut more than the current number of edges
        E = sum(len(info['downstream']) for info in self.hru_network.values())
        cuts = 0

        while self._has_cycle():
            if cuts > E:  # Ultra-safety (shouldn't trigger)
                print("    WARNING: Exceeded edge-cut bound; aborting")
                break

            comps = self._sccs()
            # Consider only cyclic components (size > 1)
            cyclic = [C for C in comps if len(C) > 1]
            if not cyclic:
                break

            for C in cyclic:
                # Find weakest edge inside this SCC
                weakest = None
                weakest_w = float('inf')

                for u in C:
                    for v in self.hru_network[u]['downstream']:
                        if v in C:
                            # Use vote weight; if missing, treat as 1
                            w = edge_counts.get((u, v), 1)
                            if w < weakest_w:
                                weakest_w = w
                                weakest = (u, v)

                if weakest is None:
                    # Fallback: break ANY edge inside the SCC
                    # This ensures we always make progress
                    for u in C:
                        if self.hru_network[u]['downstream']:
                            for v in self.hru_network[u]['downstream']:
                                if v in C:
                                    weakest = (u, v)
                                    weakest_w = edge_counts.get(weakest, 1)
                                    break
                            if weakest:
                                break

                if weakest is None:
                    print(f"    WARNING: No edge found in SCC {list(C)[:5]}")
                    continue

                u, v = weakest
                # Cut edge u->v
                self.hru_network[u]['downstream'].discard(v)
                self.hru_network[v]['upstream'].discard(u)
                cuts += 1
                print(f"    Cut edge {u} -> {v} (weight={weakest_w}) to break cycle (cut #{cuts})")

    def _find_cycle_example(self):
        """Return one simple directed cycle as a list of node IDs, or [] if none."""
        visited, stack = set(), set()
        parent = {}

        def dfs(u):
            visited.add(u)
            stack.add(u)
            for v in self.hru_network[u]['downstream']:
                if v not in visited:
                    parent[v] = u
                    result = dfs(v)
                    if result:
                        return result
                elif v in stack:
                    # reconstruct cycle from v back to v
                    cyc = []
                    x = u
                    while x != v:
                        cyc.append(x)
                        if x not in parent:
                            break
                        x = parent[x]
                    cyc.append(v)
                    cyc.reverse()
                    return cyc
            stack.remove(u)
            return None

        for u in self.hru_network:
            if u not in visited:
                parent = {u: None}
                cyc = dfs(u)
                if cyc:
                    return cyc
        return []

    def build_flow_network_from_d8(self):
        """Build network from D8 flow direction with auto-detection and edge voting"""
        print("\nBuilding flow network from D8...")

        if 'flow_dir' not in self.rasters or 'hrus' not in self.rasters:
            raise RuntimeError("Flow direction and HRU rasters required")

        flow_dir = self.rasters['flow_dir']
        hrus = self.rasters['hrus']

        # --- Detect D8 coding ---
        finite_fd = flow_dir[np.isfinite(flow_dir)].astype(int)
        unique_codes = np.unique(finite_fd)

        # Check for ESRI 1-8 style first (WhiteboxTools d8_pointer output)
        if unique_codes.size > 0 and unique_codes.min() >= 1 and unique_codes.max() <= 8:
            self._d8_offsets = self._d8_esri_1to8
            print("  Detected D8 coding: ESRI 1-8 (WhiteboxTools)")
        else:
            # Fall back to power-of-2 detection
            self._choose_d8_offsets()

        # --- Detect pointer sense (outflow vs inflow) ---
        is_outflow = self._detect_d8_direction()
        print(f"  D8 interpreted as: {'OUTFLOW' if is_outflow else 'INFLOW'} pointers")

        # --- Init nodes ---
        unique_hrus = np.unique(hrus[hrus > 0])
        for hru_id in unique_hrus:
            if np.isfinite(hru_id):
                self.hru_network[int(hru_id)] = {
                    'downstream': set(),
                    'upstream': set(),
                    'is_outlet': False,
                    'area_cells': int(np.sum(hrus == hru_id))
                }

        # --- Count pixel-level cross-HRU edges ---
        edge_counts = defaultdict(int)  # (u, v) -> count

        # Ignore bogus flow codes and zeros explicitly
        valid_fd = np.isfinite(flow_dir) & (flow_dir != 0)
        rows, cols = np.where((hrus > 0) & valid_fd)
        for r, c in zip(rows, cols):
            u0 = int(hrus[r, c])
            code = int(flow_dir[r, c])
            if code not in self._d8_offsets:
                continue
            dr, dc = self._d8_offsets[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < hrus.shape[0] and 0 <= nc < hrus.shape[1]:
                if np.isfinite(hrus[nr, nc]):
                    v0 = int(hrus[nr, nc])
                    if v0 > 0 and v0 != u0:
                        # interpret pointer sense
                        u, v = (u0, v0) if is_outflow else (v0, u0)
                        edge_counts[(u, v)] += 1

        # --- Winner-takes-all per HRU pair; FA tie-break toward higher FA ---
        fa = self.rasters.get('flow_acc', None)
        seen_pairs = set()

        def mean_fa(a):
            if fa is None:
                return 0.0
            m = (hrus == a)
            vals = fa[m]
            return float(np.nanmean(vals)) if vals.size else 0.0

        for (u, v), uv_cnt in edge_counts.items():
            if (u, v) in seen_pairs or (v, u) in seen_pairs:
                continue
            vu_cnt = edge_counts.get((v, u), 0)

            if uv_cnt > vu_cnt:
                winner = (u, v)
            elif vu_cnt > uv_cnt:
                winner = (v, u)
            else:
                # tie → direct flow toward higher FA
                mu_u = mean_fa(u)
                mu_v = mean_fa(v)
                winner = (u, v) if mu_v >= mu_u else (v, u)

            a, b = winner
            self.hru_network[a]['downstream'].add(b)
            self.hru_network[b]['upstream'].add(a)
            seen_pairs.add((u, v))
            seen_pairs.add((v, u))

        # --- Optional: Enforce single downstream per HRU ---
        if self.SINGLE_DOWNSTREAM_MODE:
            print("  Enforcing single downstream per HRU...")
            # Find strongest outgoing edge for each HRU
            best = {}
            for u, info in self.hru_network.items():
                if info['downstream']:
                    max_weight = -1
                    best_target = None
                    for v in info['downstream']:
                        weight = edge_counts.get((u, v), 1)
                        if weight > max_weight:
                            max_weight = weight
                            best_target = v
                    if best_target is not None:
                        best[u] = (best_target, max_weight)

            # Rebuild network with only strongest edges
            for nid in self.hru_network:
                self.hru_network[nid]['downstream'].clear()
                self.hru_network[nid]['upstream'].clear()

            for u, (v, weight) in best.items():
                self.hru_network[u]['downstream'].add(v)
                self.hru_network[v]['upstream'].add(u)

            print(f"    Reduced to {len(best)} single downstream connections")

        # --- Optional debug: check mutual edges (should be zero now) ---
        mutual = [(x, y) for x, info in self.hru_network.items()
                  for y in info['downstream'] if x in self.hru_network.get(y, {}).get('downstream', set())]
        if mutual:
            print(f"  WARNING: Mutual edges found: {mutual[:5]} (total {len(mutual)})")
            print("  This indicates inconsistent flow directions at HRU boundaries")

        # --- First attempt: check if we have cycles ---
        if self._has_cycle():
            print("  Cycles detected — retrying with flipped pointer sense...")

            # Clear the network
            for k in self.hru_network:
                self.hru_network[k]['downstream'].clear()
                self.hru_network[k]['upstream'].clear()

            # Rebuild with flipped sense
            seen_pairs.clear()
            for (u, v), uv_cnt in edge_counts.items():
                if (u, v) in seen_pairs or (v, u) in seen_pairs:
                    continue
                vu_cnt = edge_counts.get((v, u), 0)

                # Flip the winner logic
                if uv_cnt > vu_cnt:
                    winner = (v, u)  # flip
                elif vu_cnt > uv_cnt:
                    winner = (u, v)  # flip
                else:
                    # tie → flip tie-break direction too
                    mu_u = mean_fa(u)
                    mu_v = mean_fa(v)
                    winner = (v, u) if mu_v >= mu_u else (u, v)  # flip

                a, b = winner
                self.hru_network[a]['downstream'].add(b)
                self.hru_network[b]['upstream'].add(a)
                seen_pairs.add((u, v))
                seen_pairs.add((v, u))

            if not self._has_cycle():
                print("  ✓ Cycles resolved by flipping pointer sense.")
                is_outflow = not is_outflow  # Update the sense flag

        # --- Final cycle enforcement (handles rare HRU-vote loops) ---
        # This ALWAYS runs to guarantee a DAG, even if no cycles were detected initially
        if self._has_cycle():
            print("  Cycles remain after voting — enforcing acyclicity by trimming weakest edges...")

            # Optional: Add diagnostic for sticky triads
            example_cycle = self._find_cycle_example()
            if example_cycle and len(example_cycle) <= 5:
                print(f"  Examining cycle: {example_cycle}")
                for i in range(len(example_cycle)):
                    u = example_cycle[i]
                    v = example_cycle[(i + 1) % len(example_cycle)]
                    weight = edge_counts.get((u, v), None)
                    print(f"    Edge {u} -> {v}: weight = {weight}")

            self._force_acyclic(edge_counts)

            if self._has_cycle():
                # Something is seriously wrong if we still have cycles
                cyc = self._find_cycle_example()
                if cyc:
                    print(f"  ERROR: Example cycle still remaining: {cyc[:10]}")
                    print(f"  Full cycle length: {len(cyc)} HRUs")
                raise RuntimeError("Cycle breaker failed — check input rasters for consistency.")
            else:
                print("  ✓ Successfully enforced DAG structure by trimming weakest edges")

        # --- Mark outlets ---
        for hid, info in self.hru_network.items():
            if not info['downstream']:
                info['is_outlet'] = True

        # --- Preserve original network for later geometry heuristics ---
        self.orig_network = {k: {'downstream': v['downstream'].copy(),
                                 'upstream': v['upstream'].copy(),
                                 'is_outlet': v['is_outlet'],
                                 'area_cells': v['area_cells']} for k, v in self.hru_network.items()}

        n_outlets = sum(1 for info in self.hru_network.values() if info['is_outlet'])
        print(f"  Built network: {len(self.hru_network)} HRUs, {n_outlets} outlets")

        if n_outlets > self.MAX_OUTLETS_BEFORE_CONSOLIDATE:
            self._consolidate_outlets()

    def _consolidate_outlets(self, keep=1):
        """Consolidate multiple outlets and re-route to main outlet"""
        outlets = [hid for hid, info in self.hru_network.items() if info['is_outlet']]
        if len(outlets) <= keep:
            return

        print(f"  Consolidating {len(outlets)} outlets...")

        # Calculate contributing area for each outlet
        def contrib_area(root):
            q, vis, area = deque([root]), set(), 0
            while q:
                x = q.popleft()
                if x in vis:
                    continue
                vis.add(x)
                area += self.hru_network[x]['area_cells']
                q.extend(self.hru_network[x]['upstream'])
            return area

        areas = {o: contrib_area(o) for o in outlets}
        main = max(areas, key=areas.get)

        # Build set of HRUs that can reach main outlet
        can_reach_main = set()
        dq = deque([main])
        while dq:
            u = dq.popleft()
            if u in can_reach_main:
                continue
            can_reach_main.add(u)
            dq.extend(self.hru_network[u]['upstream'])

        # Get HRU raster for contact analysis
        hrus = self.rasters.get('hrus', None)

        # Re-route minor outlets
        for oid in outlets:
            self.hru_network[oid]['is_outlet'] = (oid == main)
            if oid == main:
                continue

            # Re-route minor outlet to neighbor that can reach main
            if not self.hru_network[oid]['downstream']:
                candidates = set(self.hru_network[oid]['upstream'])

                # Add touching neighbors if we have raster
                if hrus is not None:
                    mask = (hrus == oid)
                    boundary = mask & ~ndimage.binary_erosion(mask, structure=self._S4)

                    for nid in self.hru_network:
                        if nid == oid:
                            continue
                        nm = (hrus == nid)
                        edge = nm & ~ndimage.binary_erosion(nm, structure=self._S4)
                        if np.any(boundary & ndimage.binary_dilation(edge, structure=self._S4, iterations=1)):
                            candidates.add(nid)

                # Keep only candidates that can reach main
                candidates = [c for c in candidates if c in can_reach_main and c != oid]

                if candidates:
                    # Choose candidate with largest boundary contact
                    if hrus is not None:
                        mask = (hrus == oid)
                        boundary = mask & ~ndimage.binary_erosion(mask, structure=self._S4)

                        def contact_len(cid):
                            nm = (hrus == cid)
                            edge = nm & ~ndimage.binary_erosion(nm, structure=self._S4)
                            return int(
                                np.sum(boundary & ndimage.binary_dilation(edge, structure=self._S4, iterations=1)))

                        candidates.sort(key=contact_len, reverse=True)

                    target = candidates[0]
                    self.hru_network[oid]['downstream'].add(target)
                    self.hru_network[target]['upstream'].add(oid)
                    print(f"    Re-routed HRU {oid} -> {target}")
                else:
                    print(f"    WARNING: Could not re-route minor outlet HRU {oid}")

        print(f"  Main outlet: HRU {main}")

    def estimate_hru_geometry(self):
        """Estimate width and Dx with 4-connected morphology and caching"""
        print("\nEstimating HRU geometry...")

        if 'hrus' not in self.rasters:
            raise RuntimeError("HRU raster required")

        hrus = self.rasters['hrus']

        for idx, hru in self.hru_stats.iterrows():
            hru_id = int(hru['id'])

            # Use cached masks
            mask, boundary = self._get_hru_masks(hru_id)
            boundary_cells = np.sum(boundary)

            # Get area
            area_m2 = float(hru.get('area_m2', hru.get('area', 0)))
            if area_m2 == 0:
                raise ValueError(f"No area found for HRU {hru_id}")

            # Calculate width based on downstream contact
            if hru_id in self.orig_network:
                ds_hrus = self.orig_network[hru_id]['downstream']

                if ds_hrus:
                    total_contact = 0
                    for ds_id in ds_hrus:
                        ds_mask, ds_edge = self._get_hru_masks(ds_id)
                        # Count contact with 4-connected dilation
                        contact = np.sum(boundary & ndimage.binary_dilation(ds_edge, structure=self._S4, iterations=1))
                        total_contact += contact

                    if total_contact > 0:
                        width = total_contact * self.res[0]
                    else:
                        width = boundary_cells * self.res[0] * 0.5
                else:
                    # Outlet - use partial boundary
                    width = boundary_cells * self.res[0] * 0.7
            else:
                width = np.sqrt(area_m2)

            # Apply constraints to width
            width = max(width, self.res[0])
            width_cap = self.WIDTH_CAP_FACTOR * np.sqrt(area_m2)
            width = min(width, width_cap)

            # Calculate Dx
            Dx = max(area_m2 / width, self.res[0])

            self.hru_stats.loc[idx, 'width'] = float(width)
            self.hru_stats.loc[idx, 'Dx'] = float(Dx)

        print(f"  Geometry estimated for {len(self.hru_stats)} HRUs")

    def topological_sort_hrus(self):
        """Kahn's algorithm with deterministic main outlet selection"""
        print("\nTopological sorting HRUs (Kahn's algorithm)...")

        # Calculate in-degrees for reversed graph
        rev_graph = {nid: set() for nid in self.hru_network}
        rev_indeg = {nid: 0 for nid in self.hru_network}

        for nid, info in self.hru_network.items():
            for d in info['downstream']:
                rev_graph[d].add(nid)
                rev_indeg[nid] += 1

        # Start with outlets (zero in-degree in reversed graph)
        queue = deque([nid for nid, deg in rev_indeg.items() if deg == 0])
        sorted_hrus = []

        while queue:
            current = queue.popleft()
            sorted_hrus.append(current)

            for upstream in rev_graph[current]:
                rev_indeg[upstream] -= 1
                if rev_indeg[upstream] == 0:
                    queue.append(upstream)

        # Handle disconnected components
        if len(sorted_hrus) < len(self.hru_network):
            print(f"  WARNING: Graph has cycles or disconnected components")
            remaining = set(self.hru_network.keys()) - set(sorted_hrus)
            sorted_hrus.extend(list(remaining))

        # Create ID remapping
        self.id_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_hrus)}
        self.id_remap_inv = {v: k for k, v in self.id_remap.items()}

        # Determine main outlet by largest contributing area
        orig_outlets = [old for old, info in self.orig_network.items() if info['is_outlet']]
        if orig_outlets:
            def contrib_area_old(root):
                q, vis, area = deque([root]), set(), 0
                while q:
                    x = q.popleft()
                    if x in vis:
                        continue
                    vis.add(x)
                    area += self.orig_network[x]['area_cells']
                    q.extend(self.orig_network[x]['upstream'])
                return area

            best_old = max(orig_outlets, key=contrib_area_old)
            self.main_outlet_id = self.id_remap.get(best_old, 0)
        else:
            self.main_outlet_id = 0

        # Update stats with working ID
        self.hru_stats['wid'] = self.hru_stats['id'].map(self.id_remap)

        # Remap network
        new_network = {}
        for old_id, new_id in self.id_remap.items():
            new_network[new_id] = {
                'downstream': {self.id_remap[d] for d in self.hru_network[old_id]['downstream']},
                'upstream': {self.id_remap[u] for u in self.hru_network[old_id]['upstream']},
                'is_outlet': self.hru_network[old_id]['is_outlet'],
                'area_cells': self.hru_network[old_id]['area_cells']
            }
        self.hru_network = new_network

        # Verify ordering
        violations = [(u, v) for u, info in self.hru_network.items()
                      for v in info['downstream'] if v >= u]

        print(f"  Sorted {len(sorted_hrus)} HRUs")
        print(f"  Main outlet: HRU {self.main_outlet_id} (was {self.id_remap_inv[self.main_outlet_id]})")

        if violations:
            raise RuntimeError(f"Topological sort failed. Violations: {violations[:5]}")
        else:
            print(f"  ✓ All edges satisfy downstream < current")

    def estimate_parameters_fixed(self):
        """Estimate parameters with improved channel detection"""
        print("\nEstimating hydrological parameters...")

        for idx, hru in self.hru_stats.iterrows():
            wid = int(hru['wid'])

            # Channel detection with area threshold
            is_channel = bool(hru.get('has_channel', 0))
            if not is_channel and 'rivers' in self.rasters and 'hrus' in self.rasters:
                oid = self.id_remap_inv[wid]
                mask, _ = self._get_hru_masks(oid)
                total_pix = np.sum(mask)
                river_pix = np.sum((self.rasters['rivers'] > 0) & mask)

                # Require minimum overlap to be channel
                if river_pix / max(total_pix, 1) > self.CHANNEL_OVERLAP_MIN:
                    is_channel = True

            is_outlet = self.hru_network[wid]['is_outlet']

            # Flow velocity
            if is_channel or is_outlet:
                c_sf = 1.0  # m/s
                class_type = 'channel'
            else:
                # Auto-detect slope units
                slope_val = float(hru.get('slope_mean', 0.05))
                if slope_val > np.pi / 2:  # Likely degrees
                    slope_rad = np.deg2rad(slope_val)
                else:  # Already radians
                    slope_rad = slope_val

                slope_pct = np.tan(slope_rad) * 100

                if slope_pct > 10:
                    c_sf = 0.5
                elif slope_pct > 5:
                    c_sf = 0.3
                else:
                    c_sf = 0.1
                class_type = 'hillslope'

            # Storage parameters based on TWI
            twi = float(hru.get('twi_mean', 8.0))

            if twi > 12:  # Wet
                s_rzmax = 0.08
                m = 0.025
                t_d = 20 * 3600.0
            elif twi > 8:  # Medium
                s_rzmax = 0.06
                m = 0.015
                t_d = 50 * 3600.0
            else:  # Dry
                s_rzmax = 0.04
                m = 0.008
                t_d = 100 * 3600.0

            s_rzmax = float(np.clip(s_rzmax, 0.02, 0.08))
            t_0 = float(0.5 * t_d)

            # Geometry
            area_m2 = float(hru.get('area_m2', hru.get('area', 0)))
            width = float(hru.get('width', np.sqrt(area_m2)))
            Dx = float(hru.get('Dx', area_m2 / width))

            # Store parameters as native Python types
            self.hru_stats.loc[idx, 'class'] = class_type
            self.hru_stats.loc[idx, 'c_sf'] = float(c_sf)
            self.hru_stats.loc[idx, 's_rzmax'] = float(s_rzmax)
            self.hru_stats.loc[idx, 'm'] = float(m)
            self.hru_stats.loc[idx, 't_0'] = float(t_0)
            self.hru_stats.loc[idx, 't_d'] = float(t_d)
            self.hru_stats.loc[idx, 'area'] = float(area_m2)
            self.hru_stats.loc[idx, 'width'] = float(width)
            self.hru_stats.loc[idx, 'Dx'] = float(Dx)

        n_channels = (self.hru_stats['class'] == 'channel').sum()
        print(f"  Parameters estimated ({n_channels} channels, {len(self.hru_stats) - n_channels} hillslopes)")

    def create_flow_linkage(self):
        """Create flow linkage with improved fractions and caching"""
        print("\nCreating flow linkage...")

        linkage = []

        for wid, info in self.hru_network.items():
            if not info['downstream']:
                if not info['is_outlet']:
                    print(f"  WARNING: HRU {wid} has no downstream but is not outlet")
                continue

            to_list = sorted([int(d) for d in info['downstream']])

            # Calculate weights using cached masks
            if 'hrus' in self.rasters:
                weights = []

                oid = self.id_remap_inv[int(wid)]
                _, boundary = self._get_hru_masks(oid)

                for d in to_list:
                    ds_oid = self.id_remap_inv[d]
                    _, ds_edge = self._get_hru_masks(ds_oid)
                    contact_cells = int(np.sum(boundary & ndimage.binary_dilation(
                        ds_edge, structure=self._S4, iterations=1)))
                    weights.append(contact_cells)

                # Ensure minimum weight and normalize
                weights = np.array(weights, dtype=float)
                weights = np.maximum(weights, 1.0)
                fractions = weights / weights.sum()

                # Optional: collapse to dominant target if above threshold
                imax = int(np.argmax(fractions))
                if fractions[imax] >= self.DOMINANT_SPLIT_THRESH:
                    fractions = np.zeros(len(fractions))
                    fractions[imax] = 1.0

                fractions = fractions.tolist()
            else:
                fractions = [1.0 / len(to_list)] * len(to_list)

            linkage.append({
                'id': int(wid),
                'to': to_list,
                'frc': fractions
            })

        print(f"  Created {len(linkage)} flow connections")
        return linkage

    def create_location_file(self):
        """Create location metadata"""
        location = {
            'crs': self.crs.to_string() if self.crs else None,
            'transform': list(self.transform) if self.transform else None,
            'res': list(self.res) if self.res else None,
            'shape': list(self.shape) if self.shape else None
        }

        location_file = self.output_dir / 'location.json'
        with open(location_file, 'w') as f:
            json.dump(location, f, indent=2)

        print(f"  Created location file")

    def create_output_flux(self):
        """Create output flux for main outlet"""
        outlet_id = int(self.main_outlet_id) if self.main_outlet_id is not None else 0

        output_flux = {
            'name': ['outlet_flow', 'outlet_baseflow'],
            'id': [outlet_id, outlet_id],
            'flux': ['q_sf', 'q_sz'],
            'scale': [1.0, 1.0]
        }

        output_file = self.output_dir / 'output_flux.json'
        with open(output_file, 'w') as f:
            json.dump(output_flux, f, indent=2)

        print(f"  Created output flux for outlet HRU {outlet_id}")

    def _ensure_finite(self, df, cols):
        """Check for NaN/Inf values"""
        for c in cols:
            if c in df.columns:
                if not np.isfinite(df[c]).all():
                    bad_idx = np.where(~np.isfinite(df[c]))[0]
                    raise ValueError(f"Non-finite values in column '{c}' at indices {bad_idx[:5]}")

    def save_outputs(self):
        """Save all outputs with final validation"""
        print("\nSaving outputs...")

        # Sort by working ID
        self.hru_stats = self.hru_stats.sort_values('wid')

        # Verify contiguous IDs
        wids = self.hru_stats['wid'].astype(int).tolist()
        if sorted(wids) != list(range(len(wids))):
            raise ValueError("Working IDs must be contiguous [0..N-1]")

        # Check for NaN/Inf
        self._ensure_finite(self.hru_stats,
                            ['area', 'width', 'Dx', 'c_sf', 'm', 's_rzmax', 't_0', 't_d'])

        # Ensure positive geometry
        self.hru_stats['width'] = np.maximum(self.hru_stats['width'], float(self.res[0]))
        self.hru_stats['Dx'] = np.maximum(self.hru_stats['Dx'], float(self.res[0]))

        # Summary
        print("\n" + "=" * 60)
        print("PARAMETER SUMMARY")
        print("=" * 60)
        print(f"HRUs: {len(self.hru_stats)} | Main outlet: wid {self.main_outlet_id}")
        print(f"Total area: {self.hru_stats['area'].sum() / 1e6:.2f} km²")
        print(f"s_rzmax: {self.hru_stats['s_rzmax'].min():.3f} - {self.hru_stats['s_rzmax'].max():.3f} m")
        print(f"m: {self.hru_stats['m'].min():.4f} - {self.hru_stats['m'].max():.4f} m")
        print(f"t_0: {self.hru_stats['t_0'].min() / 3600:.1f} - {self.hru_stats['t_0'].max() / 3600:.1f} hours")
        print(f"t_d: {self.hru_stats['t_d'].min() / 3600:.1f} - {self.hru_stats['t_d'].max() / 3600:.1f} hours")
        print(f"c_sf: {self.hru_stats['c_sf'].min():.2f} - {self.hru_stats['c_sf'].max():.2f} m/s")
        print(f"width: {self.hru_stats['width'].min():.1f} - {self.hru_stats['width'].max():.1f} m")
        print(f"Dx: {self.hru_stats['Dx'].min():.1f} - {self.hru_stats['Dx'].max():.1f} m")

        n_channels = (self.hru_stats['class'] == 'channel').sum()
        channel_pct = 100 * n_channels / len(self.hru_stats)
        print(f"Channel HRUs: {n_channels} / {len(self.hru_stats)} ({channel_pct:.1f}%)")
        print("=" * 60)

        # Final graph validation
        violations = [(u, v) for u, info in self.hru_network.items()
                      for v in info['downstream'] if v >= u]
        if violations:
            raise RuntimeError(f"Ordering invalid! Examples: {violations[:5]}")

        # Class mapping (no endNode in class)
        class_mapping = {
            'hillslope': {'sf': {'type': 'cnst'}, 'sz': {'type': 'exp'}},
            'channel': {'sf': {'type': 'cnst'}, 'sz': {'type': 'exp'}}
        }

        # HRU file
        hru_data = {
            'id': self.hru_stats['wid'].astype(int).tolist(),
            'class': self.hru_stats['class'].tolist()
        }

        hru_file = self.output_dir / 'hru.json'
        with open(hru_file, 'w') as f:
            json.dump(hru_data, f, indent=2)

        # Class file
        class_file = self.output_dir / 'class.json'
        unique_classes = self.hru_stats['class'].unique()
        class_data = {cls: class_mapping.get(cls, {}) for cls in unique_classes}

        with open(class_file, 'w') as f:
            json.dump(class_data, f, indent=2)

        # Flow linkage
        linkage = self.create_flow_linkage()
        linkage_file = self.output_dir / 'flow_linkage.json'
        with open(linkage_file, 'w') as f:
            json.dump(linkage, f, indent=2)

        # Parameter files
        param_mapping = {
            'area': 'area',
            'width': 'width',
            'Dx': 'Dx',
            'c_sf': 'c_sf',
            'm': 'm',
            's_rzmax': 's_rzmax',
            't_0': 't_0',
            't_d': 't_d'
        }

        for param, filename in param_mapping.items():
            if param in self.hru_stats.columns:
                param_file = self.output_dir / f'{filename}.json'
                param_data = {
                    'id': self.hru_stats['wid'].astype(int).tolist(),
                    'value': self.hru_stats[param].astype(float).tolist()
                }
                with open(param_file, 'w') as f:
                    json.dump(param_data, f, indent=2)

        # Location and output flux
        self.create_location_file()
        self.create_output_flux()

        # Save ID mapping
        mapping_file = self.output_dir / 'id_mapping.csv'
        mapping_df = pd.DataFrame([
            {'original_id': int(old), 'new_id': int(new)}
            for old, new in self.id_remap.items()
        ])
        mapping_df.to_csv(mapping_file, index=False)

        print(f"\n  Saved all outputs to {self.output_dir}")

        # Final summary for validation
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✓ Outlets: {sum(1 for info in self.hru_network.values() if info['is_outlet'])}")
        print(f"✓ Ordering violations: 0")
        print(f"✓ Max width: {self.hru_stats['width'].max():.1f} m")
        print(f"✓ Flow connections: {len(linkage)}")
        print(f"✓ Channel percentage: {channel_pct:.1f}%")
        print("=" * 60)

    def run_conversion(self):
        """Run complete conversion pipeline"""
        print("\n" + "=" * 60)
        print("DYNATOPGIS CONVERSION - PRODUCTION VERSION WITH CYCLE FIX")
        print("=" * 60)

        try:
            self.load_data()
            self.build_flow_network_from_d8()
            self.estimate_hru_geometry()
            self.topological_sort_hrus()
            self.estimate_parameters_fixed()
            self.save_outputs()

            print("\n" + "=" * 60)
            print("✓ CONVERSION COMPLETE!")
            print(f"Outputs saved to: {self.output_dir}")
            print("=" * 60)

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function"""
    # Configuration - DIRECT INPUT FOR IDE
    input_dir = "D:\\filled_dem"  # Your DEM processing output
    output_dir = "D:\\dynatopgis_output"  # DynatopGIS files

    # Run conversion
    converter = DynatopGISConverter(input_dir, output_dir)
    converter.run_conversion()


if __name__ == "__main__":
    main()
