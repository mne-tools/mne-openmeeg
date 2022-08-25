import mne
from mne.datasets import sample
import matplotlib.pyplot as plt

print(__doc__)

data_path = sample.data_path()

raw_fname = data_path / 'MEG/sample/sample_audvis_raw.fif'
# fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fwd_fname = 'openmeeg_eeg-fwd.fif'

subjects_dir = data_path / 'subjects'

# Read the forward solutions with surface orientation
fwd = mne.read_forward_solution(fwd_fname)
leadfield = fwd['sol']['data']
print("Leadfield size : %d x %d" % leadfield.shape)

###############################################################################
# Compute sensitivity maps
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

eeg_map.plot(time_label='Gradiometer sensitivity', subjects_dir=subjects_dir,
             clim=dict(lims=[0, 50, 100]))
