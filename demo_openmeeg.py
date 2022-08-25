# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>

import numpy as np
import mne
from mne.transforms import _ensure_trans, invert_transform, Transform
from mne.transforms import combine_transforms, apply_trans
from mne.surface import transform_surface_to
from mne.forward._make_forward import _to_forward_dict
from mne.io.constants import FIFF

import openmeeg as om


def swap_faces(faces):
    return faces[:, [1, 0, 2]]


def _prepare_trans(info, trans, coord_frame='head'):
    head_mri_t = _ensure_trans(trans, 'head', 'mri')
    dev_head_t = info['dev_head_t']
    del trans

    # Figure out our transformations
    if coord_frame == 'meg':
        head_trans = invert_transform(dev_head_t)
        meg_trans = Transform('meg', 'meg')
        mri_trans = invert_transform(combine_transforms(
            dev_head_t, head_mri_t, 'meg', 'mri'))
    elif coord_frame == 'mri':
        head_trans = head_mri_t
        meg_trans = combine_transforms(dev_head_t, head_mri_t, 'meg', 'mri')
        mri_trans = Transform('mri', 'mri')
    else:  # coord_frame == 'head'
        head_trans = Transform('head', 'head')
        meg_trans = info['dev_head_t']
        mri_trans = invert_transform(head_mri_t)
    return head_trans, meg_trans, mri_trans


def _convert_bem_surf(surf, mri_trans, head_trans):
    """Write bem model to .geom file."""
    coord_frame = 'head'
    surf = transform_surface_to(surf, coord_frame,
                                [mri_trans, head_trans], copy=True)
    return surf


def get_dipoles(src, mri_trans, head_trans):
    """Write dipole locations aka source space."""
    src_rr = np.r_[src[0]['rr'][src[0]['inuse'].astype(bool)],
                   src[1]['rr'][src[1]['inuse'].astype(bool)]]
    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI:
        src_rr = apply_trans(mri_trans, src_rr)
        # src_nn = apply_trans(mri_trans, src_nn, move=False)
    elif src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
        src_rr = apply_trans(head_trans, src_rr)
        # src_nn = apply_trans(head_trans, src_nn, move=False)

    pos = src_rr
    ori = np.kron(np.ones(len(pos))[:, None], np.eye(3))
    pos = np.kron(pos.T, np.ones(3)[None, :]).T
    return pos, ori


def _make_forward(eeg_leadfield, ch_names, info, src, trans_fname):
    fwd = eeg_leadfield.astype(np.float32).T
    fwd = _to_forward_dict(fwd, ch_names)
    picks = mne.pick_channels(info['ch_names'], ch_names)
    fwd['info'] = mne.pick_info(info, picks)
    with fwd["info"]._unlock():
        fwd['info']['mri_file'] = trans_fname
        fwd['info']['mri_id'] = fwd['info']['file_id']
        fwd['mri_head_t'] = invert_transform(mne.read_trans(trans_fname))
        fwd['info']['mri_head_t'] = fwd['mri_head_t']
        fwd['info']['meas_file'] = ""
    fwd['src'] = src
    fwd['surf_ori'] = False
    return fwd


def make_forward_solution(info, trans_fname, src, bem_model, meg=True,
                          eeg=True, mindist=0.0, ignore_ref=False, n_jobs=1,
                          verbose=None):
    assert not meg  # XXX for now

    conductivity = [s['sigma'] for s in bem_model]
    coord_frame = 'head'
    trans = mne.read_trans(trans_fname)
    head_trans, meg_trans, mri_trans = _prepare_trans(info, trans, coord_frame)

    # OpenMEEG
    meshes = []
    for surf in bem_model[::-1]:
        surf = _convert_bem_surf(surf, mri_trans, head_trans)
        meshes.append(
            (surf['rr'], swap_faces(surf['tris']))
        )

    geom = om.make_nested_geometry(meshes, conductivity)

    assert geom.is_nested()
    assert geom.selfCheck()

    pos, ori = get_dipoles(src, mri_trans, head_trans)
    dipoles = np.c_[pos, ori]
    dipoles = om.Matrix(np.asfortranarray(dipoles))

    eeg_picks = mne.pick_types(info, meg=False, eeg=True, ref_meg=False)
    eeg_loc = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
    eeg_loc = apply_trans(head_trans, eeg_loc)
    ch_names = [info['ch_names'][k] for k in eeg_picks]

    hm = om.HeadMat(geom)
    hm.invert()
    hminv = hm
    dsm = om.DipSourceMat(geom, dipoles, "Brain")

    # For EEG
    eeg_sensors = om.Sensors(om.Matrix(np.asfortranarray(eeg_loc)), geom)

    h2em = om.Head2EEGMat(geom, eeg_sensors)
    eeg_leadfield = om.GainEEG(hminv, dsm, h2em)

    fwd = _make_forward(eeg_leadfield.array(), ch_names, info, src, trans_fname)

    return fwd


if __name__ == '__main__':
    from mne.datasets import sample
    data_path = sample.data_path()

    # the raw file containing the channel location + types
    raw_fname = data_path / 'MEG/sample/sample_audvis_raw.fif'
    # The paths to Freesurfer reconstructions
    subjects_dir = data_path / 'subjects'
    subject = 'sample'
    # The transformation file obtained by coregistration
    trans_fname = str(data_path / 'MEG/sample/sample_audvis_raw-trans.fif')

    info = mne.io.read_info(raw_fname)
    src = mne.setup_source_space(subject, spacing='ico3',
                                 subjects_dir=subjects_dir, add_dist=False)

    # conductivity = (0.3,)  # for single layer
    conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(subject='sample', ico=3,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)

    fwd = make_forward_solution(info, trans_fname, src, model, eeg=True,
                                meg=False)
    mne.write_forward_solution('openmeeg_eeg-fwd.fif', fwd, overwrite=True)
