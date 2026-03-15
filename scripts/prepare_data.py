"""
数据准备脚本

用法：原始数据放在 data/raw/<dataset>/（ligand/、protein/、index.csv），脚本只读不改；
organize 把「能成功复制的」条目写入 data/processed/<dataset>/（cleaned/ + index.csv），后续预处理与训练都在 processed 下。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INDEX_ID_COL = "Concatenated ID"
INDEX_AFFINITY_COL = "Log Binding Affinity"


# ----- organize -----
def _copy_with_lowercase_sdf_header(src: Path, dest: Path) -> None:
    """复制文件到 dest；若为 .sdf 则首行写为小写后写入，否则直接复制。不修改 src。"""
    import shutil

    if src.suffix.lower() == ".sdf":
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if lines:
            lines[0] = lines[0].rstrip("\n").lower() + ("\n" if lines[0].endswith("\n") else "\n")
        with open(dest, "w", encoding="utf-8") as f:
            f.writelines(lines)
    else:
        shutil.copy2(src, dest)


def cmd_organize(args: argparse.Namespace) -> None:
    """从 raw 只读，在 processed 下生成新文件。不删除、不修改 raw 下任何内容。"""
    import shutil

    raw_root = Path(args.raw_root).resolve()
    target_root = Path(args.target_root).resolve()
    index_file = Path(args.index_file).resolve()
    # 保证不删、不改 raw：仅从 raw 读，只写 target_root 下
    assert raw_root != target_root, "raw 与 processed 必须不同目录"
    assert index_file.parent == raw_root, "index 须在 raw 目录下"
    target_subdir = target_root / "cleaned"
    target_subdir.mkdir(parents=True, exist_ok=True)

    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    df = pd.read_csv(index_file)
    if INDEX_ID_COL not in df.columns or INDEX_AFFINITY_COL not in df.columns:
        raise ValueError(
            f"index.csv 必须包含表头: {INDEX_ID_COL}, {INDEX_AFFINITY_COL}。当前: {df.columns.tolist()}"
        )
    df = df.rename(columns={INDEX_ID_COL: "pdb_id", INDEX_AFFINITY_COL: "affinity"})

    initial_len = len(df)
    df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")
    no_affinity_mask = df["affinity"].isna()
    no_affinity_ids = df.loc[no_affinity_mask, "pdb_id"].astype(str).str.strip().str.lower().tolist()
    df = df.dropna(subset=["affinity"])
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning(
            "Dropped %d rows with missing or invalid affinity (first 20): %s",
            dropped,
            no_affinity_ids[:20],
        )
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()

    ligand_dir = raw_root / "ligand"
    protein_dir = raw_root / "protein"
    pdb_ids = sorted(df["pdb_id"].astype(str))
    success_count = 0
    missing_count = 0
    success_ids: list[str] = []

    for pdb_id in tqdm(pdb_ids):
        pdb_id = pdb_id.lower()
        src_ligand = ligand_dir / f"{pdb_id}_ligand.sdf"
        if not src_ligand.exists():
            src_ligand = ligand_dir / f"{pdb_id}_ligand.mol2"
        src_protein = protein_dir / f"{pdb_id}_protein.pdb"
        src_esm = protein_dir / f"{pdb_id}_esm.npz"

        if not src_ligand.exists() or not src_protein.exists():
            missing_count += 1
            continue

        target_pdb_dir = target_subdir / pdb_id
        target_pdb_dir.mkdir(parents=True, exist_ok=True)
        try:
            # 仅写入 target_root 下，绝不写 raw
            dest_ligand = target_pdb_dir / (f"{pdb_id}_ligand{src_ligand.suffix}".lower())
            assert str(dest_ligand.resolve()).startswith(str(target_root)), "dest 必须在 processed 下"
            _copy_with_lowercase_sdf_header(src_ligand, dest_ligand)
            dest_protein = target_pdb_dir / f"{pdb_id}_protein.pdb"
            shutil.copy2(src_protein, dest_protein)
            if src_esm.exists():
                shutil.copy2(src_esm, target_pdb_dir / f"{pdb_id}_esm.npz")
            success_count += 1
            success_ids.append(pdb_id)
        except Exception as e:
            logger.error("Error copying %s: %s", pdb_id, e)

    target_index = target_root / "index.csv"
    df_out = df[df["pdb_id"].isin(success_ids)][["pdb_id", "affinity"]].copy()
    df_out = df_out.rename(columns={"pdb_id": INDEX_ID_COL, "affinity": INDEX_AFFINITY_COL})
    df_out.to_csv(target_index, index=False)
    logger.info("Success: %d, Missing: %d. Output: %s", success_count, missing_count, target_root)


STEP_ORDER = ("organize",)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="数据准备：data/raw/<dataset>/ 下须有 ligand/、protein/、index.csv；结果写入 data/processed/<dataset>/。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="数据集名称（与 data/raw/<dataset>/、data/processed/<dataset>/ 对应），如 hiqbind、pdbbind",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="organize",
        help="步骤，或 all；当前仅支持 organize（复制到 processed 并做小写文件名与 SDF 首行）",
    )

    args = parser.parse_args()
    dataset = args.dataset.strip().lower()
    if not dataset:
        parser.error("dataset 不能为空")

    raw_root = PROJECT_ROOT / "data" / "raw" / dataset
    target_root = PROJECT_ROOT / "data" / "processed" / dataset

    if args.steps.strip().lower() == "all":
        steps_to_run = list(STEP_ORDER)
    else:
        steps_to_run = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
        for s in steps_to_run:
            if s not in STEP_ORDER:
                parser.error(f"未知步骤: {s}，可选: {','.join(STEP_ORDER)}")

    index_csv = raw_root / "index.csv"

    if "organize" in steps_to_run:
        if not raw_root.exists():
            parser.error(f"原始数据目录不存在: {raw_root}，请将 ligand/、protein/、index.csv 放在该目录下")
        for sub in ("ligand", "protein"):
            if not (raw_root / sub).is_dir():
                parser.error(f"缺少目录 {raw_root / sub}，data/raw/<dataset>/ 下须包含 ligand/ 与 protein/")
        if not index_csv.exists():
            parser.error(f"索引文件不存在: {index_csv}（表头须为 Concatenated ID, Log Binding Affinity）")

    for step in steps_to_run:
        logger.info("===== %s =====", step)
        if step == "organize":
            cmd_organize(argparse.Namespace(
                raw_root=str(raw_root),
                target_root=str(target_root),
                index_file=str(index_csv),
            ))
    logger.info("Done. Ran steps: %s", ", ".join(steps_to_run))


if __name__ == "__main__":
    main()
