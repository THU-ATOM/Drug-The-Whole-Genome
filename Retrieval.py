###########################################################################################
# DrugClip Max Similarity Filter - Retrieval
# Author: Wenjin Liu
# This program is distributed under the Apache License. See NOTICE and LICENSE files for details.
# Date: 2025-09-02
###########################################################################################
# OpenVsynthes/Filter/DrugClipFilterUtils/Retrieval.py
import os
import sys
import subprocess

class Retrieval:
    """
    Run DrugClip retrieval by invoking DrugClip_Retrieval.sh.
    Stateless: all paths/flags are passed in or set via configure(...).
    """

    def __init__(self, ligands_dir: str | None = None, protein_dir: str | None = None,
                 cache_name: str | None = None, dir_utils: str | None = None,
                 working_dir: str | None = None, project_name: str | None = None,
                 save_dir: str | None = None, embed_dir: str | None = None) -> None:
        self.ligands_dir = ligands_dir
        self.protein_dir = protein_dir
        self.cache_name = cache_name
        self.dir_utils = dir_utils
        self.working_dir = working_dir
        self.project_name = project_name
        self.save_dir = save_dir
        self.embed_dir = embed_dir

        self.lmdb_mol = None
        self.lmdb_pocket = None
        self.script_sh = None
        self._is_configured = False

        if all([ligands_dir, protein_dir, cache_name, dir_utils, working_dir, project_name]):
            self.configure(ligands_dir, protein_dir, cache_name, dir_utils, working_dir, project_name,
                           save_dir=save_dir, embed_dir=embed_dir)

    def configure(self, ligands_dir: str, protein_dir: str, cache_name: str, dir_utils: str,
                  working_dir: str, project_name: str, save_dir: str | None = None,
                  embed_dir: str | None = None) -> None:
        self.ligands_dir = ligands_dir
        self.protein_dir = protein_dir
        self.cache_name = cache_name
        self.dir_utils = dir_utils
        self.working_dir = working_dir
        self.project_name = project_name

        self.lmdb_mol = os.path.join(self.ligands_dir, "ligands.lmdb")
        self.lmdb_pocket = os.path.join(self.protein_dir, "pocket.lmdb")
        self.script_sh = os.path.join(self.dir_utils, "DrugClip_Retrieval.sh")

        assert os.path.isdir(self.ligands_dir), f"[DrugClipFilter] ligands_dir not found: {self.ligands_dir}"
        assert os.path.isdir(self.protein_dir), f"[DrugClipFilter] protein_dir not found: {self.protein_dir}"
        assert os.path.exists(self.lmdb_mol), f"[DrugClipFilter] ligands LMDB missing: {self.lmdb_mol}"
        assert os.path.exists(self.lmdb_pocket), f"[DrugClipFilter] pocket LMDB missing: {self.lmdb_pocket}"
        assert os.path.exists(self.script_sh), f"[DrugClipFilter] script not found: {self.script_sh}"

        if save_dir is None:
            save_dir = os.path.join(working_dir, "DockingData", project_name, "Input", "DrugClip", "Result")
        if embed_dir is None:
            embed_dir = os.path.join(working_dir, "DockingData", project_name, "Input", "DrugClip", "Embed")
        self.save_dir = save_dir
        self.embed_dir = embed_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.embed_dir, exist_ok=True)

        self._is_configured = True

    def run_retrieval(self, ligands_dir: str | None = None, protein_dir: str | None = None,
                      cache_name: str | None = None, dir_utils: str | None = None,
                      working_dir: str | None = None, project_name: str | None = None,
                      save_dir: str | None = None, embed_dir: str | None = None,
                      topk_percent: float = 100.0, fold_version: str = "6_folds",
                      multi_conf: bool = False, use_cache: bool = False,
                      cuda_devices: str = "0", log_file: str | None = None) -> None:
        if not self._is_configured:
            required = [ligands_dir, protein_dir, cache_name, dir_utils, working_dir, project_name]
            assert all(required), (
                "[DrugClipFilter] Retrieval is not configured. Either call `configure(...)` first, "
                "or pass all path arguments to `run_retrieval(...)`."
            )
            self.configure(ligands_dir, protein_dir, cache_name, dir_utils, working_dir, project_name,
                           save_dir=save_dir, embed_dir=embed_dir)

        self._run_retrieval_impl(topk_percent, fold_version, multi_conf, use_cache, cuda_devices, log_file)

    def _run_retrieval_impl(self, topk_percent: float, fold_version: str, multi_conf: bool,
                            use_cache: bool, cuda_devices: str, log_file: str | None) -> None:
        u_str = "True" if use_cache else "False"
        x_str = "True" if multi_conf else "False"

        cmd = [
            # "bash", "DrugClip_Retrieval.sh",
            "bash", self.script_sh, 
            "-m", self.lmdb_mol,
            "-n", self.cache_name,
            "-p", self.lmdb_pocket,
            "-S", self.save_dir,
            "-D", self.embed_dir,
            "-u", u_str,
            "-t", str(topk_percent),
            "-f", fold_version,
            "-x", x_str,
            "-g", cuda_devices,
        ]
        if log_file:
            cmd += ["-l", log_file]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        # Give the script the exact python used by your kernel
        env["PYTHONBIN"] = sys.executable
        # If script does `python ...`, ensure it resolves to current env
        # (optional safety) env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]

        print("[DrugClipFilter] Launching DrugClip retrieval:")
        print(" ", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)
        print(f"[DrugClipFilter] Retrieval finished.\n  Save:  {self.save_dir}\n  Embed: {self.embed_dir}")
