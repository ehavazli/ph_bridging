"""
Adapted from https://github.com/insarlab/MintPy/blob/main/src/mintpy/objects/conncomp.py
"""

import itertools
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csgraph as csg
from scipy.spatial import cKDTree
from skimage import measure
from skimage import morphology as morph
from skimage import segmentation as seg

from .ramp import deramp


def label_conn_comp(
    mask: np.ndarray,
    min_area: float = 2.5e3,
    erosion_size: int = 5,
    print_msg: bool = False,
) -> Tuple[np.ndarray, int]:
    """Label / clean up the connected components mask."""
    label_img, num_label = measure.label(mask, connectivity=1, return_num=True)
    min_area = min(min_area, label_img.size * 3e-3)
    if print_msg:
        print(f"Removing regions with area < {int(min_area)}")
    mask = morph.remove_small_objects(label_img, min_size=min_area, connectivity=1)
    label_img[mask == 0] = 0
    label_img, num_label = measure.label(label_img, connectivity=1, return_num=True)
    if erosion_size > 0:
        erosion_structure = np.ones((erosion_size, erosion_size))
        label_erosion_img = morph.erosion(label_img, erosion_structure).astype(np.uint8)
        erosion_regions = measure.regionprops(label_erosion_img)
        if len(erosion_regions) < num_label:
            if print_msg:
                print("Regions lost during morphological erosion operation:")
            label_erosion = [reg.label for reg in erosion_regions]
            for orig_reg in measure.regionprops(label_img):
                if orig_reg.label not in label_erosion:
                    label_img[label_img == orig_reg.label] = 0
                    if print_msg:
                        print(
                            f"Label: {orig_reg.label}, Area: {orig_reg.area}, Bbox: {orig_reg.bbox}"
                        )
        label_img, num_label = measure.label(label_img, connectivity=1, return_num=True)
    return label_img, num_label


def label_boundary(
    label_img: np.ndarray,
    num_label: int,
    erosion_size: int = 5,
    print_msg: bool = False,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Label the boundary of the labeled array."""
    if erosion_size > 0:
        erosion_structure = np.ones((erosion_size, erosion_size))
        label_erosion_img = morph.erosion(label_img, erosion_structure).astype(np.uint8)
        erosion_regions = measure.regionprops(label_erosion_img)
        if len(erosion_regions) < num_label:
            if print_msg:
                print("Regions lost during morphological erosion operation:")
            label_erosion = [reg.label for reg in erosion_regions]
            for orig_reg in measure.regionprops(label_img):
                if orig_reg.label not in label_erosion:
                    label_img[label_img == orig_reg.label] = 0
                    if print_msg:
                        print(
                            f"Label: {orig_reg.label}, Area: {orig_reg.area}, Bbox: {orig_reg.bbox}"
                        )
    label_img, num_label = measure.label(label_img, connectivity=1, return_num=True)
    label_bound = seg.find_boundaries(label_erosion_img, mode="thick").astype(np.uint8)
    label_bound *= label_erosion_img
    return label_img, num_label, label_bound


class connectComponent:
    """Object for bridging connected components."""

    def __init__(self, conncomp: np.ndarray, metadata: dict):
        """Initialize the ConnectComponent object."""
        if not isinstance(conncomp, np.ndarray):
            raise ValueError("Input conncomp is not np.ndarray")
        self.conncomp = conncomp
        self.metadata = metadata
        self.refY = int(metadata.get("REF_Y", -1))
        self.refX = int(metadata.get("REF_X", -1))
        self.length, self.width = self.conncomp.shape

    def label(
        self, min_area: float = 2.5e3, erosion_size: int = 5, print_msg: bool = False
    ) -> None:
        """Label the connected components."""
        self.labelImg, self.numLabel = label_conn_comp(
            self.conncomp, min_area=min_area, print_msg=print_msg
        )
        self.labelImg, self.numLabel, self.labelBound = label_boundary(
            self.labelImg, self.numLabel, erosion_size=erosion_size, print_msg=print_msg
        )
        if self.refY == -1 or self.refX == -1:
            regions = measure.regionprops(self.labelImg)
            idx = np.argmax([region.area for region in regions])
            self.labelRef = regions[idx].label
        else:
            self.labelRef = self.labelImg[self.refY, self.refX]

    def get_all_bridge(self) -> Tuple[Dict[str, Any], np.ndarray]:
        """Search all possible connections among labeled regions."""
        trees = [
            cKDTree(region.coords) for region in measure.regionprops(self.labelBound)
        ]
        self.connDict = dict()
        self.distMat = np.zeros((self.numLabel, self.numLabel), dtype=np.float32)
        for i, j in itertools.combinations(range(self.numLabel), 2):
            dist, idx = trees[i].query(measure.regionprops(self.labelBound)[j].coords)
            idx_min = np.argmin(dist)
            yxj = measure.regionprops(self.labelBound)[j].coords[idx_min, :]
            yxi = measure.regionprops(self.labelBound)[i].coords[idx[idx_min], :]
            dist_min = dist[idx_min]
            n0, n1 = str(i + 1), str(j + 1)
            conn = dict()
            conn[n0] = yxi
            conn[n1] = yxj
            conn["distance"] = dist_min
            self.connDict[f"{n0}_{n1}"] = conn
            self.distMat[i, j] = self.distMat[j, i] = dist_min
        return self.connDict, self.distMat

    def find_mst_bridge(self) -> List[Dict[str, Union[int, float]]]:
        """Search for bridges to connect all labeled areas using the minimum spanning tree algorithm."""
        if not hasattr(self, "distMat"):
            self.get_all_bridge()
        distMatMst = csg.minimum_spanning_tree(self.distMat)
        succs, preds = csg.breadth_first_order(
            distMatMst, i_start=self.labelRef - 1, directed=False
        )
        self.bridges = []
        for i in range(1, succs.size):
            n0 = preds[succs[i]] + 1
            n1 = succs[i] + 1
            if n0 > n1:
                nn = [str(n1), str(n0)]
            else:
                nn = [str(n0), str(n1)]
            conn = self.connDict[f"{nn[0]}_{nn[1]}"]
            y0, x0 = conn[str(n0)]
            y1, x1 = conn[str(n1)]
            bridge = dict()
            bridge["x0"] = x0
            bridge["y0"] = y0
            bridge["x1"] = x1
            bridge["y1"] = y1
            bridge["label0"] = n0
            bridge["label1"] = n1
            bridge["distance"] = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            self.bridges.append(bridge)
        self.num_bridge = len(self.bridges)
        return self.bridges

    def get_bridge_endpoint_aoi_mask(
        self, bridge: Dict[str, int], radius: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get AOI mask for bridge endpoints."""
        x0, y0 = bridge["x0"], bridge["y0"]
        x1, y1 = bridge["x1"], bridge["y1"]
        x00 = max(0, x0 - radius)
        x01 = min(self.width, x0 + radius)
        y00 = max(0, y0 - radius)
        y01 = min(self.length, y0 + radius)
        x10 = max(0, x1 - radius)
        x11 = min(self.width, x1 + radius)
        y10 = max(0, y1 - radius)
        y11 = min(self.length, y1 + radius)
        aoi_mask0 = np.zeros(self.labelImg.shape, dtype=np.bool_)
        aoi_mask1 = np.zeros(self.labelImg.shape, dtype=np.bool_)
        aoi_mask0[y00:y01, x00:x01] = True
        aoi_mask1[y10:y11, x10:x11] = True
        return aoi_mask0, aoi_mask1

    def unwrap_conn_comp(
        self,
        unw: np.ndarray,
        radius: int = 50,
        ramp_type: Optional[str] = None,
        print_msg: bool = False,
    ) -> np.ndarray:
        """Do bridging."""
        start_time = time.time()
        radius = int(min(radius, min(self.conncomp.shape) * 0.05))
        unw = np.array(unw, dtype=np.float32)
        if self.refY != -1 and self.refX != -1:
            unw[unw != 0.0] -= unw[self.refY, self.refX]
        if ramp_type is not None:
            if print_msg:
                print(f"Estimating a {ramp_type} ramp")
            ramp_mask = self.labelImg == self.labelRef
            unw, ramp = deramp(unw, ramp_mask, ramp_type, metadata=self.metadata)
        for bridge in self.bridges:
            aoi_mask0, aoi_mask1 = self.get_bridge_endpoint_aoi_mask(
                bridge, radius=radius
            )
            label_mask0 = self.labelImg == bridge["label0"]
            label_mask1 = self.labelImg == bridge["label1"]
            value0 = np.nanmedian(unw[aoi_mask0 * label_mask0])
            value1 = np.nanmedian(unw[aoi_mask1 * label_mask1])
            diff_value = value1 - value0
            num_jump = (np.abs(diff_value) + np.pi) // (2.0 * np.pi)
            if diff_value > 0:
                num_jump *= -1
            unw[label_mask1] += 2.0 * np.pi * num_jump
            if print_msg:
                print(
                    f"Phase diff {bridge['label1']}_{bridge['label0']}: {diff_value:04.1f} rad --> Num of jump: {num_jump}"
                )
        if ramp_type is not None:
            unw += ramp
        if print_msg:
            print(f"Time used: {time.time()-start_time:.2f} secs.")
        return unw

    def plot_bridge(self, ax, cmap: str = "jet", radius: int = 50) -> None:
        """Plot bridges on the provided axis."""
        ax.imshow(self.labelImg, cmap=cmap, interpolation="nearest")
        for bridge in self.bridges:
            ax.plot(
                [bridge["x0"], bridge["x1"]], [bridge["y0"], bridge["y1"]], "w-", lw=1
            )
            if radius > 0:
                aoi_mask0, aoi_mask1 = self.get_bridge_endpoint_aoi_mask(
                    bridge, radius=radius
                )
                label_mask0 = self.labelImg == bridge["label0"]
                label_mask1 = self.labelImg == bridge["label1"]
                ax.plot(
                    np.nonzero(aoi_mask0 * label_mask0)[1],
                    np.nonzero(aoi_mask0 * label_mask0)[0],
                    "gray",
                    alpha=0.3,
                )
                ax.plot(
                    np.nonzero(aoi_mask1 * label_mask1)[1],
                    np.nonzero(aoi_mask1 * label_mask1)[0],
                    "gray",
                    alpha=0.3,
                )
