import os
import math
import gc

import numpy as np
import torch
from tqdm import tqdm

from ..trainer import predict


def read_gps_file(path):
    """
    DenseUAV GPS format:

        path E120.x N30.x height

    返回：
        gps[path_without_ext] = (lon, lat)
    """

    gps = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                continue

            rel_path = parts[0]

            try:
                lon = float(
                    parts[1].replace("E", "")
                )

                lat = float(
                    parts[2].replace("N", "")
                )

            except ValueError:
                continue

            gps[rel_path] = (
                lon,
                lat
            )

    return gps


def haversine(
    lon1,
    lat1,
    lon2,
    lat2
):
    """
    Haversine distance in meters.
    """

    R = 6371000.0

    lon1 = math.radians(lon1)
    lat1 = math.radians(lat1)

    lon2 = math.radians(lon2)
    lat2 = math.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2.0) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        *
        R
        *
        math.asin(
            math.sqrt(a)
        )
    )


def build_gallery_gps(
    gps_all,
    gallery_root
):
    """
    为 gallery 的 3033 个 sampling IDs
    建立：

        gallery_id -> (lon, lat)
    """

    result = {}

    for sid in sorted(
        os.listdir(gallery_root)
    ):

        folder = os.path.join(
            gallery_root,
            sid
        )

        if not os.path.isdir(folder):
            continue

        prefixes = [
            f"train/satellite/{sid}/",
            f"test/satellite/{sid}/",
            f"test/gallery_satellite/{sid}/"
        ]

        coord = None

        # GPS_ALL 可能来自 train 或 test satellite。
        for key, value in gps_all.items():

            if any(
                    key.startswith(p)
                    for p in prefixes
            ):
                coord = value
                break

        if coord is not None:
            result[sid] = coord

    return result


def build_query_gps(
    gps_test,
    query_root
):

    result = {}

    for sid in sorted(os.listdir(query_root)):

        folder = os.path.join(
            query_root,
            sid
        )

        if not os.path.isdir(folder):
            continue

        prefix = f"test/satellite/{sid}/"

        coord = None

        for key,value in gps_test.items():

            if key.startswith(prefix):
                coord = value
                break

        if coord is not None:
            result[sid]=coord


    return result


def nearest_gallery_ids(
    query_gps,
    gallery_gps
):
    """
    对每个 query location，
    找距离最近的 gallery location。
    """

    gallery_ids = list(
        gallery_gps.keys()
    )

    gallery_xy = np.array(
        [
            gallery_gps[sid]
            for sid in gallery_ids
        ],
        dtype=np.float64
    )

    result = {}

    for qid, (
        qlon,
        qlat
    ) in query_gps.items():

        distances = []

        for glon, glat in gallery_xy:

            distances.append(
                haversine(
                    qlon,
                    qlat,
                    glon,
                    glat
                )
            )

        distances = np.asarray(
            distances
        )

        min_dist = distances.min()

        nearest = [
            gallery_ids[i]
            for i, d in enumerate(distances)
            if abs(d - min_dist) < 1e-6
        ]

        result[qid] = {
            "gallery_ids": nearest,
            "distance_m": float(min_dist)
        }

    return result

def aggregate_features(features, ids):
    """
    DenseUAV:
    多张图片属于同一个sampling location
    需要按照ID平均池化
    """

    from collections import defaultdict

    feature_dict = defaultdict(list)

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    elif isinstance(ids, tuple):
        ids = list(ids)


    for feat, sid in zip(features, ids):

        if isinstance(sid, torch.Tensor):
            sid = str(sid.item())

        else:
            sid = str(sid)

        feature_dict[sid].append(feat)


    agg_features = []
    agg_ids = []


    for sid, feat_list in feature_dict.items():

        feat = torch.stack(
            feat_list,
            dim=0
        ).mean(
            dim=0
        )

        feat = torch.nn.functional.normalize(
            feat,
            dim=0
        )

        agg_features.append(feat)
        agg_ids.append(sid)


    agg_features = torch.stack(
        agg_features,
        dim=0
    )


    return agg_features, agg_ids

def compute_ap(
    ranking,
    positive_indices
):
    """
    Standard AP.
    """

    positive_indices = set(
        positive_indices
    )

    if len(
        positive_indices
    ) == 0:
        return 0.0

    hit = 0
    ap = 0.0

    for rank, index in enumerate(
        ranking,
        start=1
    ):

        if index in positive_indices:

            hit += 1

            ap += (
                hit
                /
                float(rank)
            )

    return (
        ap
        /
        len(positive_indices)
    )


def evaluate(
    config,
    model,
    query_loader,
    gallery_loader,
    query_root,
    gallery_root,
    gps_test_file,
    gps_all_file,
    ranks=(1, 5, 10),
    cleanup=True,
    step=0
):

    print()
    print("=" * 70)
    print("DenseUAV V2 Evaluation")
    print("=" * 70)

    print(
        "Extracting gallery features..."
    )

    gallery_features, gallery_ids = predict(
        config,
        model,
        gallery_loader
    )

    print(
        "Extracting query features..."
    )

    query_features, query_ids = predict(
        config,
        model,
        query_loader
    )

    print("DEBUG query_ids:")
    print(query_ids[:10])
    print(type(query_ids[0]))

    # ---------------------------------------------------------
    # Feature aggregation
    # ---------------------------------------------------------

    gallery_features, gallery_ids = aggregate_features(
        gallery_features,
        gallery_ids
    )

    query_features, query_ids = aggregate_features(
        query_features,
        query_ids
    )

    print(
        "Query locations :",
        len(query_ids)
    )

    print(
        "Gallery locations:",
        len(gallery_ids)
    )

    # ---------------------------------------------------------
    # GPS
    # ---------------------------------------------------------

    print(
        "Loading DenseUAV GPS..."
    )

    gps_test = read_gps_file(
        gps_test_file
    )

    gps_all = read_gps_file(
        gps_all_file
    )

    query_gps = build_query_gps(
        gps_test,
        query_root
    )

    gallery_gps = build_gallery_gps(
        gps_all,
        gallery_root
    )

    print("DEBUG query_gps:")
    print(list(query_gps.keys())[:10])
    print(len(query_gps))

    print(
        "Query GPS      :",
        len(query_gps)
    )

    print(
        "Gallery GPS    :",
        len(gallery_gps)
    )

    # ---------------------------------------------------------
    # nearest GT
    # ---------------------------------------------------------

    gt = nearest_gallery_ids(
        query_gps,
        gallery_gps
    )

    gallery_id_to_index = {
        sid: i
        for i, sid in enumerate(
            gallery_ids
        )
    }

    # ---------------------------------------------------------
    # Similarity
    # ---------------------------------------------------------

    similarity = (
        query_features
        @
        gallery_features.T
    )

    similarity = (
        similarity
        .detach()
        .cpu()
        .numpy()
    )

    recalls = {
        k: []
        for k in ranks
    }

    aps = []

    valid_queries = 0

    gt_distances = []

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    for qi, qid in enumerate(
        query_ids
    ):

        if qid not in gt:
            continue

        positive_gallery_ids = (
            gt[qid]["gallery_ids"]
        )

        positive_indices = [
            gallery_id_to_index[sid]
            for sid in positive_gallery_ids
            if sid in gallery_id_to_index
        ]

        if len(
            positive_indices
        ) == 0:
            continue

        valid_queries += 1

        score = similarity[qi]

        ranking = np.argsort(
            -score
        )

        # Recall
        for k in ranks:

            topk = ranking[:k]

            hit = any(
                x in positive_indices
                for x in topk
            )

            recalls[k].append(
                1.0 if hit else 0.0
            )

        # AP
        aps.append(
            compute_ap(
                ranking,
                positive_indices
            )
        )

        gt_distances.append(
            gt[qid]["distance_m"]
        )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("DenseUAV V2 Results")
    print("=" * 70)

    print(
        "Valid queries:",
        valid_queries,
        "/",
        len(query_ids)
    )

    for k in ranks:

        value = (
            np.mean(
                recalls[k]
            )
            * 100.0
            if recalls[k]
            else 0.0
        )

        print(
            f"Recall@{k}: "
            f"{value:.4f}%"
        )

    map_value = (
        np.mean(aps)
        * 100.0
        if aps
        else 0.0
    )

    print(
        f"mAP: {map_value:.4f}%"
    )

    if gt_distances:

        print(
            "Mean nearest-gallery "
            "GPS distance: "
            f"{np.mean(gt_distances):.3f} m"
        )

        print(
            "Max nearest-gallery "
            "GPS distance: "
            f"{np.max(gt_distances):.3f} m"
        )

    print("=" * 70)

    if cleanup:

        del (
            query_features,
            gallery_features,
            similarity
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "R1":
            np.mean(recalls[1]) * 100
            if recalls[1]
            else 0.0,

        "R5":
            np.mean(recalls[5]) * 100
            if recalls[5]
            else 0.0,

        "R10":
            np.mean(recalls[10]) * 100
            if recalls[10]
            else 0.0,

        "AP":
            map_value
    }