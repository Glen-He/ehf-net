"""
数据整理命令入口。

负责把 raw 数据集整理为 processed 目录结构，
并校验索引文件与基础文件布局是否完整。
"""


import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

INDEX_ID_COL = "Concatenated ID"
INDEX_AFFINITY_COL = "Log Binding Affinity"


# ----- 数据整理步骤 -----
def _copy_with_lowercase_sdf_header(src: Path, dest: Path) -> None:
    """
    复制文件到 dest；若为 .sdf 则首行写为小写后写入，否则直接复制，不修改 src。
    """
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
    """
    整理原始数据到 processed 目录。

    只读取 `raw` 目录中的源文件并在 `processed` 下生成新文件，
    不会修改或删除原始数据，适合作为首次整理入口。

    Args:
        args: 命令行解析后的参数对象，包含当前命令所需的配置。

    Raises:
        FileNotFoundError: 当依赖文件不存在时抛出。
        ValueError: 当输入参数或运行时状态不满足要求时抛出。
    """
    raw_root = Path(args.raw_root).resolve()
    target_root = Path(args.target_root).resolve()
    index_file = Path(args.index_file).resolve()
    # 保证不删、不改原始目录：仅从 `raw_root` 读取，只写入 `target_root`。
    if raw_root == target_root:
        raise ValueError("raw_root and target_root must be different directories")
    if index_file.parent != raw_root:
        raise ValueError("index file must be located under raw_root")
    target_subdir = target_root / "cleaned"
    target_subdir.mkdir(parents=True, exist_ok=True)

    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    df = pd.read_csv(index_file)
    if INDEX_ID_COL not in df.columns or INDEX_AFFINITY_COL not in df.columns:
        raise ValueError(
            f"index.csv must contain the headers: {INDEX_ID_COL}, {INDEX_AFFINITY_COL}. "
            f"Current columns: {df.columns.tolist()}"
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
            # 仅写入 `target_root`，绝不回写原始目录。
            dest_ligand = target_pdb_dir / (f"{pdb_id}_ligand{src_ligand.suffix}".lower())
            if not dest_ligand.resolve().is_relative_to(target_root):
                raise RuntimeError("destination path must stay under target_root")
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
    """
    数据整理入口函数。

    负责解析数据集名称和整理步骤，校验 raw 目录结构，
    并将原始数据复制整理到统一的 processed 目录布局。
    """
    parser = argparse.ArgumentParser(
        description="Prepare processed data from data/raw/<dataset>/ into data/processed/<dataset>/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name mapped to data/raw/<dataset>/ and data/processed/<dataset>/, e.g. hiqbind or pdbbind",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="organize",
        help="Step name or 'all'. Currently only 'organize' is supported.",
    )

    args = parser.parse_args()
    dataset = args.dataset.strip().lower()
    if not dataset:
        parser.error("dataset must not be empty")

    raw_root = PROJECT_ROOT / "data" / "raw" / dataset
    target_root = PROJECT_ROOT / "data" / "processed" / dataset

    if args.steps.strip().lower() == "all":
        steps_to_run = list(STEP_ORDER)
    else:
        steps_to_run = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
        for s in steps_to_run:
            if s not in STEP_ORDER:
                parser.error(f"Unknown step: {s}. Available steps: {','.join(STEP_ORDER)}")

    index_csv = raw_root / "index.csv"

    if "organize" in steps_to_run:
        if not raw_root.exists():
            parser.error(
                f"Raw data directory not found: {raw_root}. "
                "Expected ligand/, protein/, and index.csv under this directory."
            )
        for sub in ("ligand", "protein"):
            if not (raw_root / sub).is_dir():
                parser.error(
                    f"Missing directory: {raw_root / sub}. "
                    "data/raw/<dataset>/ must contain both ligand/ and protein/."
                )
        if not index_csv.exists():
            parser.error(
                f"Index file not found: {index_csv}. "
                "Required headers: Concatenated ID, Log Binding Affinity."
            )

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
