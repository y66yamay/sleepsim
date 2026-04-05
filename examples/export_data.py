"""Export synthetic PSG data to disk in various formats.

Demonstrates:
1. Saving a full dataset (NPZ + CSV hypnograms + traits.csv + metadata.json)
2. Saving individual subjects to NPZ
3. Loading a subject back from NPZ
4. (Optional) Saving to EDF if pyedflib is installed
"""

from pathlib import Path

from sleepsim import (
    SleepDataGenerator,
    save_subject_npz, load_subject_npz,
    save_hypnogram_csv, save_traits_csv,
)


def main():
    out_dir = Path("examples/output_data")

    print("=" * 60)
    print("PSG data export examples")
    print("=" * 60)

    # Small dataset for demonstration
    gen = SleepDataGenerator(
        n_subjects=3, sampling_rate=128, duration_hours=1.0, seed=42,
        condition="healthy",
    )

    # --- 1. Save the whole dataset ---
    print("\n[1] Saving full dataset via SleepDataGenerator.save_to_disk() ...")
    summary = gen.save_to_disk(out_dir / "healthy_npz", fmt="npz")
    print(f"    Saved {summary['n_subjects_saved']} subjects to "
          f"{summary['output_dir']}")

    # Show what was written
    print("\n    Directory contents:")
    for p in sorted((out_dir / "healthy_npz").rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            rel = p.relative_to(out_dir / "healthy_npz")
            print(f"      {rel}  ({size_kb:.1f} KB)")

    # --- 2. Save individual subjects ---
    print("\n[2] Saving individual subjects with save_subject_npz() ...")
    indiv_dir = out_dir / "individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)
    gen2 = SleepDataGenerator(
        n_subjects=2, sampling_rate=128, duration_hours=0.5, seed=7,
        condition="rbd",
    )
    for data in gen2.generate_subject_iter():
        sid = data["traits"].subject_id
        save_subject_npz(data, indiv_dir / f"rbd_subject_{sid:03d}.npz")
        save_hypnogram_csv(
            data["hypnogram"],
            indiv_dir / f"rbd_subject_{sid:03d}_hypnogram.csv",
            epoch_sec=gen2.epoch_sec,
        )
        print(f"    Saved subject {sid}")

    # --- 3. Load a subject back ---
    print("\n[3] Loading subject back with load_subject_npz() ...")
    loaded = load_subject_npz(indiv_dir / "rbd_subject_000.npz")
    print(f"    Loaded subject {loaded['traits'].subject_id} "
          f"({loaded['traits'].condition})")
    print(f"    PSG shape: {loaded['psg_data'].shape}")
    print(f"    Hypnogram shape: {loaded['hypnogram'].shape}")
    print(f"    FC matrix shape: {loaded['fc_matrix'].shape}")
    print(f"    Sampling rate: {loaded['sampling_rate']} Hz")

    # --- 4. Trait table only ---
    print("\n[4] Saving a trait table only with save_traits_csv() ...")
    save_traits_csv(gen.subjects, out_dir / "trait_table.csv")
    print(f"    Wrote {out_dir / 'trait_table.csv'}")

    # --- 5. EDF (optional) ---
    print("\n[5] Attempting EDF export (requires pyedflib) ...")
    try:
        import pyedflib  # noqa: F401
        gen3 = SleepDataGenerator(
            n_subjects=1, sampling_rate=128, duration_hours=0.5, seed=1,
            condition="osa",
        )
        summary_edf = gen3.save_to_disk(out_dir / "osa_edf", fmt="edf")
        print(f"    Saved {summary_edf['n_subjects_saved']} subject(s) as EDF "
              f"to {summary_edf['output_dir']}")
    except ImportError:
        print("    pyedflib not installed. Skipping EDF export.")
        print("    Install with: pip install pyedflib")

    print("\nDone!")


if __name__ == "__main__":
    main()
