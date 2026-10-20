"""Core module defining the main LocalizationSession class.

This module provides the central orchestration class for running a source
localization experiment. It integrates all other components (mesh, data, models,
etc.) and exposes a high-level API for session management.
"""
import os
# Set the JAX environment variable to disable pre-allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Standard library imports
import copy
import gc
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import h5py
import shutil
import jax
import numpy as np
import torch
import array

# Local application imports
from ._internal import _utils
from .data_handler import ExperimentData
from .efield import EFieldModel
from .mesh import CorticalMesh
from .model import ResponseModel
from .optimizer import StimulusOptimizer
from .plotting import Visualizer

# Type Aliases for clarity
Tensor = torch.Tensor


class LocalizationSession:
    """Orchestrates a source localization experiment by managing all components.

    This class holds instances of various components (mesh, data, models,
    plotting) and directs the flow of the experiment.

    Attributes:
        device (torch.device): PyTorch device for computation ('cpu' or 'cuda').
        save_path (str): Base path for saving experiment data.
        save_dir (str): Specific directory for the current session's saved files.
        stimulator (Optional[Dict[str, Any]]): Stimulator coil parameters.
        source (Optional[Dict[str, Any]]): Simulated source parameters.
        mesh (CorticalMesh): The cortical mesh component.
        data_handler (ExperimentData): The trial data management component.
        efield_model (EFieldModel): The E-field physics model component.
        response_model (ResponseModel): The statistical response model component.
        optimizer (StimulusOptimizer): The stimulus selection component.
        visualizer (Visualizer): The plotting and visualization component.
        silent_mode (bool): If True, disables printouts and automatic file I/O.
        session_ended (bool): Flag indicating if the session has ended.
        trial_completed (bool): Flag indicating if the current trial was completed.
    """

    def __init__(self,
                 cortex: Dict[str, Tensor],
                 save_path: str,
                 stimulator: Optional[Dict[str, Any]] = None,
                 efield_set: Optional[Tensor] = None,
                 ROI: Optional[Tensor] = None,
                 device: Optional[str] = None,
                 silent_mode: Optional[bool] = False):
        """Initializes the LocalizationSession and all its components.

        Args:
            cortex: Dictionary with 'vertices' and 'faces' tensors for the cortical mesh.
            save_path: Base directory path for saving experiment data.
            stimulator: Dictionary of stimulator specifications.
            efield_set: Tensor of E-field maps, with a shape of (n_coils, n_vertices, 3).
            ROI: Tensor of indices defining the region of interest on the cortex.
            device: Computing device, 'cpu' or 'cuda'. If None, it is chosen automatically.
            silent_mode: If True, disables printouts, automatic saving, and plotting.
        """
        _utils.set_random_seed()

        # Basic Setup
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not silent_mode:
            print('Torch device:', device)
        self.device = torch.device(device)
        self.save_path = save_path
        self.save_dir = ""
        self.silent_mode = silent_mode
        self.stimulator = self._move_stimulator_to_device(stimulator)
        self.source: Optional[Dict[str, Any]] = None

        # Instantiate Components
        self.mesh = CorticalMesh(cortex, ROI)
        self.data_handler = ExperimentData(self.device)
        self.efield_model = EFieldModel(efield_set, self.mesh, self.device)
        self.response_model = ResponseModel(self.mesh, self.device)
        self.optimizer = StimulusOptimizer(self.efield_model, self.response_model, self.stimulator, self.device) if (self.stimulator is not None and efield_set is not None) else None
        self.visualizer = Visualizer()
        self.session_ended = False
        self.trial_completed = False

        if self.optimizer and not silent_mode:
            print(f"JAX is configured to run on: {self.optimizer.jax_device.platform.upper()}")

        if not silent_mode:
            if efield_set is None or self.stimulator is None:
                print("Stimulator or E-field model not specified, stimulus optimization disabled.")

        self.initialize_trials()

    @classmethod
    def load_experiment(cls, save_dir: str, device: Optional[str] = None, silent_mode: Optional[bool] = False) -> 'LocalizationSession':
        """Loads a saved experiment from a directory.

        This method creates a new save directory to avoid overwriting the
        original experiment file upon modification.

        Args:
            save_dir: The directory containing the 'experiment_data.hdf5' file.
            device: Computing device ('cpu' or 'cuda'). If None, chosen automatically.
            silent_mode: If True, disables printouts, automatic saving, and plotting.

        Returns:
            A LocalizationSession instance initialized with the saved state.

        Raises:
            ValueError: If the loaded data is missing the 'cortex' key.
        """
        experiment_data_file = os.path.join(save_dir, 'experiment_data.hdf5')
        data_dict = _utils.load_from_hdf5(experiment_data_file)

        if 'cortex' not in data_dict:
            raise ValueError("Missing required key: 'cortex'.")

        # Get optional data
        stimulator = data_dict.get('stimulator')
        efield_set = data_dict.get('efield_set')
        ROI = data_dict.get('ROI')

        if efield_set is not None:
            efield_set = efield_set.float()

        # Basic initialization
        cortex = data_dict['cortex']

        if 'efield_trials' in data_dict:
            data_dict['efield_trials'] = data_dict['efield_trials'].float()

        # Save data from loaded session inside a new directory
        save_path = os.path.join(os.path.dirname(save_dir), 'loaded_sessions')
        instance = cls(
            cortex=cortex,
            save_path=save_path,
            stimulator=stimulator,
            efield_set=efield_set,
            ROI=ROI,
            device=device,
            silent_mode=silent_mode
        )

        # Overwrite the fresh experiment file with the original file
        # This includes the e_mags and angles for all trials, required for the response function probing functionality
        if not silent_mode:
            new_filepath = os.path.join(instance.save_dir, 'experiment_data.hdf5')
            try:
                shutil.copy(experiment_data_file, new_filepath)
            except Exception as e:
                print(f"Error copying original save file to new directory: {e}")

        # Load experiment data to the fresh instance
        instance.data_handler.load_data(data_dict)
        if 'source' in data_dict:
            instance.source = data_dict['source']

        # Load model parameters and update source probabilities (~= calling the "localize" method)
        instance.response_model.load_data(data_dict)
        if 'informed_prior' in data_dict:
            print("Informed prior in use.")
            prior_source_prob = data_dict['informed_prior'].to(instance.device)
            instance.response_model.prior_log_source_prob = instance.response_model._safe_log(prior_source_prob)
            instance.response_model.log_source_prob = copy.deepcopy(instance.response_model.prior_log_source_prob)
        instance.update_source_probabilities()

        # Plot
        if not silent_mode:
            instance.plot(re_initialize=True)

        return instance

    # Public API Methods
    @property
    def trial_number(self):
        return self.data_handler.trial_number

    def start_plotter(self):
        """Launches the interactive Dash application for visualization."""
        self.visualizer.start_plotter()

    def stop_plotter(self):
        """Stops the running Dash server process."""
        self.visualizer.stop_plotter()

    def initialize_trials(self):
        """Initializes data structures for a new session and (re)starts the plotter."""
        self._set_save_dir()
        self.visualizer.save_dir = self.save_dir
        self.data_handler.initialize_trials(len(self.mesh.ROI))
        self.response_model.initialize_response_function_parameters()
        self.response_model.initialize_probabilities()
        _utils.set_random_seed()

        if not self.silent_mode:
            self._create_save_file()
            self.visualizer.stop_plotter()
            self.visualizer.start_plotter()
            self.plot(re_initialize=True)

    def __del__(self):
        """Destructor to ensure session resources are released."""
        self.end_session()

    def end_session(self):
        """Ends the session, releasing resources and shutting down the plotter."""
        if self.session_ended:
            return

        self.visualizer.stop_plotter()
        gc.collect()
        jax.clear_caches()
        if self.device.type == 'cuda':
            try:
                torch.cuda.empty_cache()
            except AttributeError:
                # Raised when __del__ runs during interpreter shutdown, after torch.cuda is torn down
                pass
        if not self.silent_mode:
            print("Session ended.")
            
        self.session_ended = True

    def get_active_responses(self) -> Tensor:
        """Returns recorded responses from the active trial buffer."""
        return self.data_handler.get_active_responses()

    def get_active_efield_trials(self) -> Tensor:
        """Returns recorded E-field trials from the active trial buffer."""
        return self.data_handler.get_active_efield_trials()

    def start_trial(self):
        """Manages trial state and checks for an source prior."""
        if self.trial_number == 0:
            self._check_for_informed_prior()
            self.data_handler.advance_trial_counter()
        else:
            if self.trial_completed:
                self.data_handler.advance_trial_counter()
            else:
                print("Resuming a failed trial...")
        self.trial_completed = False

    def complete_trial(self):
        """Marks the current trial as completed."""
        self.trial_completed = True

    def record_response_trial(self, response: bool):
        """Stores the response of the current trial."""
        self.data_handler.record_response_trial(response)

    def simulate_response(self):
        """Simulates a response for the current trial based on the source model.

        Raises:
            RuntimeError: If a source has not been defined.
        """
        if not self.source:
            raise RuntimeError("Cannot simulate response without a defined source.")

        efield = self.data_handler.efield_trials[self.trial_number - 1].unsqueeze(0)
        e_mag, angles = self.efield_model.efield_to_mag_and_angle(efield)

        source_indices = self.source['index']
        source_params = self.source['rf_params']

        # Compare random values to response probabilities to simulate outcome
        rand_vals = torch.rand((1, len(source_indices)))
        probs = self.response_model.response_func(e_mag[:, source_indices].cpu(), angles[:, source_indices].cpu(), source_params)
        response = torch.any(probs.view(1, -1) > rand_vals)

        self.record_response_trial(response.item())

    def generate_source(self, source_indices_full_cortex: List[int]):
        """Generates a simulated source at given mesh nodes with random parameters.
                
        Args:
            source_indices_full_cortex: List of vertex indices on the full, original mesh.
        """
        n_sources = len(source_indices_full_cortex)

        # Generate randomized response function parameters for the source
        true_rf_translate = (torch.rand(n_sources) + 0.5) * 100
        true_rf_param_dict = {
            'rf_slope': {'values': (torch.rand(n_sources) + 0.5) / 8},
            'rf_translate': {'values': true_rf_translate},
            'rf_fourier1': {'values': (torch.rand(n_sources) / 2 - 0.25) * true_rf_translate},
            'rf_fourier2': {'values': (torch.rand(n_sources) / 2 - 0.25) * true_rf_translate},
            'rf_fourier3': {'values': (torch.rand(n_sources) / 2 - 0.25) * true_rf_translate},
            'rf_fourier4': {'values': (torch.rand(n_sources) / 2 - 0.25) * true_rf_translate},
        }

        source_indices_roi = [_utils.mesh_ind_to_ROI_ind(idx, self.mesh.ROI) for idx in source_indices_full_cortex]
        self.source = {'index': source_indices_roi, 'rf_params': true_rf_param_dict}

        if not self.silent_mode:
            self._create_save_file()  # Re-save to include the new source
            self.plot(re_initialize=True)

    def localize(self, full: bool = False):
        """Updates the model based on all data collected so far.

        This involves fitting the response model and then updating the
        posterior source probabilities.

        Args:
            full: If True, fits model parameters with more optimization
                steps for a final, more thorough localization. If False, uses
                fewer steps, which is recommended during an active measurement
                to prevent overfitting to preliminary data.
        """
        if full:
            # Initialize with MLE. Temporarily increase optimization steps for a more thorough fit
            default_opt_steps = self.response_model.opt_settings['general']['opt_steps']
            self.response_model.opt_settings['general']['opt_steps'] = 2000
            self.fit_model_to_data()
            self.response_model.opt_settings['general']['opt_steps'] = default_opt_steps
            
            # Run MCMC sampling to get full posterior distributions over parameters
            self.response_model.fit_distribution_model_to_data(self.data_handler, self.efield_model)
            self.response_model.update_distributional_source_probabilities(self.data_handler, self.efield_model)
        else:
            self.fit_model_to_data()

        if not self.silent_mode:
            self.save()
            self.plot(re_initialize=False)

    def choose_next_stimulus(self) -> Tensor:
        """Optimizes the E-field for the next stimulus to maximize information gain.

        Returns:
            Tensor of current slopes for the optimal stimulus.

        """
        if self.optimizer is None:
            raise ValueError("E-field optimization requires an optimizer.")

        didt = self.optimizer.choose_next_stimulus(self.response_model)

        return didt.cpu()
        
    def record_efield_trial(self, current_slopes: Any):
        """Calculates E-field from current slopes and records it for the current trial.
        
        Args:
            current_slopes: Coil current slopes. Can be a torch.Tensor (from Python) or a 
                     list/np.ndarray (from a MATLAB interface).
        """
        # Convert to Tensor if necessary and send to torch device
        if not isinstance(current_slopes, torch.Tensor):
            current_slopes = torch.tensor(list(current_slopes), dtype=torch.float32)
        current_slopes = current_slopes.to(self.device)

        next_E = self._didt_to_efield(current_slopes)
        self.data_handler._record_efield_trial(next_E)

    def fit_model_to_data(self):
        """Fits the response function model to the accumulated trial data."""
        self.response_model.fit_model_to_data(self.data_handler, self.efield_model)
        self.response_model.update_source_probabilities(self.data_handler, self.efield_model)

    def update_source_probabilities(self):
        """Computes the Bayesian posterior update of source probabilities."""
        self.response_model.update_source_probabilities(self.data_handler, self.efield_model)

    def import_trial_data(self, efield_trials: Tensor, responses: Tensor):
        """Imports previous trial data into the session.

        Args:
            efield_trials: Tensor of E-field values, with a shape of (n_trials, n_vertices, 3).
            responses: List or tensor of recorded responses as boolean or binary values.
        """
        data_dict = {'efield_trials': efield_trials.float(), 'responses': responses.float()}
        self.data_handler.load_data(data_dict)
        # If E-field basis was not loaded, compute it now from the imported data
        if any(basis is None for basis in [self.efield_model.efield_basis, self.efield_model.null_dir_basis, self.efield_model.efield_basis_means]):
            self.efield_model.apply_pca(self.data_handler.efield_trials)

    def save_figure(self, filename: str, scale: int = 2):
        """Saves the current Plotly figure as a PDF image."""
        plot_data_package = self._gather_plot_data()
        self.visualizer.save_figure(filename, scale, plot_data_package)

    def plot(self, re_initialize=True):
        """Gathers data and injects it into the visualizer to create/update the plot."""
        plot_data_package = self._gather_plot_data()
        self.visualizer.plot(plot_data_package, re_initialize)

    def save(self):
        """Saves session variables and appends any missing trial data to the HDF5 file."""
        # Generate save directory if it wasn't set (e.g., due to silent_mode)
        if not self.save_dir:
            time_string = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.save_dir = os.path.join(self.save_path, time_string)
            os.makedirs(self.save_dir, exist_ok=True)

        filepath = os.path.join(self.save_dir, 'experiment_data.hdf5')
        
        # Ensure the base file and required datasets exist before appending
        needs_initialization = not os.path.exists(filepath)
        if not needs_initialization:
            try:
                with h5py.File(filepath, 'r') as f:
                    if 'efield_trials' not in f:
                        needs_initialization = True
            except OSError:
                needs_initialization = True
                
        if needs_initialization:
            # Temporarily bypass silent_mode to force file structure creation
            original_silent_mode = self.silent_mode
            self.silent_mode = False
            self._create_save_file()
            self.silent_mode = original_silent_mode

        try:
            with h5py.File(filepath, 'a') as f:
                # Overwrite model parameters (latest fit)
                data_to_overwrite = {
                    'param_dict': {k: {'values': v['values']} for k, v in self.response_model.param_dict.items()},
                    'posterior_samples': self.response_model.posterior_samples
                }
                for key, data in data_to_overwrite.items():
                    self._var_to_hdf5(f, key, data)

                # Determine if new trial data is available
                current_file_size = f['efield_trials'].shape[0]
                total_trials = self.trial_number
                if current_file_size >= total_trials:
                    return  # No new trial data to save

                # Extract the missing slice
                delta_efields = self.data_handler.efield_trials[current_file_size:total_trials]
                delta_responses = self.data_handler.responses[current_file_size:total_trials]

                delta_mags, delta_angles = self.efield_model.efield_to_mag_and_angle(delta_efields)

                self._append_to_hdf5_dataset(f, 'efield_trials', delta_efields)
                self._append_to_hdf5_dataset(f, 'responses', delta_responses)
                self._append_to_hdf5_dataset(f, 'e_mags', delta_mags)
                self._append_to_hdf5_dataset(f, 'angles', delta_angles)

        except Exception as e:
            print(f"Error saving to file '{filepath}': {e}")

    # --- Private Helper Methods ---

    def _move_stimulator_to_device(self, stimulator):
        if stimulator:
            stimulator_on_device = {}
            for key, value in stimulator.items():

                # Fix data type if loaded from hdf5
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    value = value.decode('utf-8')

                # Check if the value is a tensor and move it to the device
                if isinstance(value, torch.Tensor):
                    stimulator_on_device[key] = value.to(self.device)
                # Otherwise, just assign the value as-is
                else:
                    stimulator_on_device[key] = value
        else:
            stimulator_on_device = None
        return stimulator_on_device
    
    def _didt_to_efield(self, current_slopes: Tensor):
        """Calculates the total E-field from current slopes.
        
        Args:
            current_slopes: Coil current slopes.

        Returns:
            Total E-field. 
        """
        efield = torch.sum((self.efield_model.efield_set * current_slopes.view(-1, 1, 1)), 0)
        return efield


    def _gather_plot_data(self) -> Dict[str, Any]:
        """Collects all necessary data for the visualizer into a single dictionary."""
        source_prob = self.response_model._safe_exp(self.response_model.log_source_prob)

        optimizer_losses = self.optimizer.losses if self.optimizer else torch.tensor([0])

        if self.trial_number > 0:
            latest_efield = self.data_handler.efield_trials[self.trial_number - 1, :, :]
            active_efields = self.get_active_efield_trials()
            e_mags, angles = self.efield_model.efield_to_mag_and_angle(active_efields)
            active_responses = self.data_handler.get_active_responses()
        else:
            # Provide zero-filled tensors if no trials have run yet
            latest_efield = torch.zeros_like(self.data_handler.efield_trials[0])
            e_mags = torch.zeros(1, self.data_handler.efield_trials.shape[1])
            angles = torch.zeros(1, self.data_handler.efield_trials.shape[1])
            active_responses = torch.zeros(1)

        return {
            # Static Mesh Info
            'mesh_vertices': self.mesh.vertices,
            'mesh_faces': self.mesh.faces,
            'mesh_ROI': self.mesh.ROI,
            'mesh_coarse_inds': self.mesh.coarse_mesh_inds,
            'mesh_camera': self.mesh.camera,
            'efield_basis': self.efield_model.efield_basis,
            'null_dir_basis': self.efield_model.null_dir_basis,
            # Dynamic Data
            'source_prob': source_prob,
            'latest_efield': latest_efield,
            'optimizer_losses': optimizer_losses,
            'response_model_losses': self.response_model.losses,
            'param_dict': self.response_model.param_dict,
            'trial_number': self.trial_number,
            'active_responses': active_responses,
            'e_mags_cpu': e_mags.cpu(),
            'angles_cpu': angles.cpu(),
            # Source info (can change)
            'source': self.source,
        }

    def _set_save_dir(self):
        """Creates a unique, timestamped save directory."""
        if self.silent_mode:
            return
        time_string = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.save_dir = os.path.join(self.save_path, time_string)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Data will be saved to: {self.save_dir}")

    def _create_save_file(self):
        """Creates the initial HDF5 save file.

        This file stores static information and initializes empty, resizable
        datasets for dynamic trial-by-trial data.
        """
        if self.silent_mode:
            return
        filepath = os.path.join(self.save_dir, 'experiment_data.hdf5')
        with h5py.File(filepath, 'w') as f:
            # Save static data first
            self._var_to_hdf5(f, 'ROI', self.mesh.ROI)
            self._var_to_hdf5(f, 'cortex', {'vertices': self.mesh.vertices, 'faces': self.mesh.faces, 'vertex_normals': self.mesh.vertex_normals})
            if self.efield_model.efield_set is not None:
                self._var_to_hdf5(f, 'efield_set', self.efield_model.efield_set)
                self._var_to_hdf5(f, 'efield_subspace', {
                    'efield_basis': self.efield_model.efield_basis,
                    'null_dir_basis': self.efield_model.null_dir_basis,
                    'efield_basis_means': self.efield_model.efield_basis_means
                })
            if self.stimulator is not None:
                self._var_to_hdf5(f, 'stimulator', self.stimulator)
            if self.source is not None:
                self._var_to_hdf5(f, 'source', self.source)

            # Initialize empty, resizable datasets for dynamic trial data
            n_vertices = len(self.mesh.ROI) if self.mesh.ROI is not None else self.mesh.vertices.shape[0]
            # chunks=True is recommended for resizable datasets for better performance
            f.create_dataset('efield_trials', (0, n_vertices, 3), maxshape=(None, n_vertices, 3), dtype='f4', chunks=True)
            f.create_dataset('responses', (0,), maxshape=(None,), dtype='f4', chunks=True)
            f.create_dataset('e_mags', (0, n_vertices), maxshape=(None, n_vertices), dtype='f4', chunks=True)
            f.create_dataset('angles', (0, n_vertices), maxshape=(None, n_vertices), dtype='f4', chunks=True)

    def _append_to_hdf5_dataset(self, h5_file: h5py.File, dataset_path: str, data_slice: Any):
        """Appends a slice of data to a resizable HDF5 dataset."""
        if isinstance(data_slice, torch.Tensor):
            data_slice = data_slice.cpu().numpy()
        
        # Ensure data is in a NumPy array format for consistent handling
        data_slice = np.asarray(data_slice)

        dataset = h5_file[dataset_path]
        current_size = dataset.shape[0]
        
        # Ensure data_slice has a batch dimension if it's missing
        if data_slice.ndim == len(dataset.shape) - 1:
            data_slice = np.expand_dims(data_slice, axis=0)
        
        # Resize the dataset and append the new data
        new_size = current_size + data_slice.shape[0]
        dataset.resize(new_size, axis=0)
        dataset[current_size:new_size] = data_slice

    def _var_to_hdf5(self, h5_group: h5py.Group, path: str, data: Any):
        """Recursively saves dictionaries and tensors to an HDF5 group.

        This method is for overwriting data, suitable for model parameters or static info.
        Posterior samples are compressed with gzip for space efficiency.
        """
        if isinstance(data, dict):
            for key, item in data.items():
                self._var_to_hdf5(h5_group, f"{path}/{key}", item)
        else:
            # This logic overwrites the dataset if it exists.
            if path in h5_group:
                del h5_group[path]

            # Prepare data
            if isinstance(data, torch.Tensor):
                data_array = data.cpu().numpy()
            else:
                data_array = data

            # Apply gzip compression only to posterior_samples (large arrays)
            use_compression = '/posterior_samples' in path and isinstance(data_array, np.ndarray) and data_array.shape

            if use_compression:
                h5_group.create_dataset(path, data=data_array, compression='gzip', compression_opts=4)
            else:
                h5_group.create_dataset(path, data=data_array)

    def _check_for_informed_prior(self):
        """Checks for and applies an informed prior from the save file.

        An informed prior allows starting the experiment with a non-uniform
        distribution of source probabilities.
        """
        if self.silent_mode:
            return
        filepath = os.path.join(self.save_dir, 'experiment_data.hdf5')
        try:
            with h5py.File(filepath, 'r') as f:
                if 'informed_prior' in f:
                    print("Informed prior in use.")
                    informed_prior = f['informed_prior'][()]
                    prior_source_prob = torch.tensor(informed_prior, device=self.device)
                    self.response_model.prior_log_source_prob = self.response_model._safe_log(prior_source_prob)
                    self.response_model.log_source_prob = copy.deepcopy(self.response_model.prior_log_source_prob)
        except (FileNotFoundError, KeyError):
            # It's okay if the file or key doesn't exist; we just use the default prior.
            pass
