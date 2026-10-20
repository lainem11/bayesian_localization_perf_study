from ABL import LocalizationSession, preprocessing
import os
from pathlib import Path

def make_session(seed):
    def _find_repo_root():
        for p in [Path.cwd(), *Path.cwd().parents]:
            if (p / 'pyproject.toml').exists():
                return str(p)
        return str(Path.cwd())

    wrkdir = _find_repo_root()

    # Locate the bundled example data (E-field set and cortex mesh)
    example_model = os.path.join(wrkdir, "examples", "saved", "example_subject", "example_model.pkl")
    E, _, cortex = preprocessing.load_example_model(example_model)

    # Where to save this session's outputs
    save_path = os.path.join(wrkdir, "examples", "saved", "example_subject")

    # Specify stimulator with pre-defined hardware constraints
    stimulator_path = os.path.join(wrkdir, "examples", "coils", "5coil_example.json")
    stimulator = preprocessing.get_stimulator_from_file(stimulator_path, constraint_type='5coil_strain')

    # Start localization instance
    device = 'cuda'
    S = LocalizationSession(cortex=cortex, save_path=save_path, stimulator=stimulator, efield_set=E, device=device, silent_mode=True)

    # Add source point for simulating responses
    source_ind = [seed]
    S.generate_source(source_ind)

    return S

# Obtain shapes
def get_tensor_info(tensor):
    return f"Shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}"

if __name__ == "__main__":
    seed=2679
    S = make_session(seed)
    
    print(f"efield_set; {get_tensor_info(S.efield_model.efield_set)}, size: {S.efield_model.efield_set.element_size() * S.efield_model.efield_set.numel() / (1024 ** 2):.2f} MB.<br>")
    print(f"efield_basis; {get_tensor_info(S.efield_model.efield_basis)}.<br>")
    print(f"efield_null_dir_basis; {get_tensor_info(S.efield_model.null_dir_basis)}.<br>")
    print(f"efield_trials; {get_tensor_info(S.data_handler.efield_trials)}.<br>")
    print(f"ROI length: {len(S.mesh.ROI)}.<br>")
    print(f"Number of coils: {S.optimizer.n_coils}.<br>")