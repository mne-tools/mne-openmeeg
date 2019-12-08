# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>

import numpy as np

import mne
from mne.transforms import _ensure_trans, invert_transform, Transform
from mne.transforms import combine_transforms, apply_trans
from mne.surface import transform_surface_to
from mne.forward._make_forward import _to_forward_dict
from mne.io.constants import FIFF

import openmeeg as om


def write_tri(fname, points, faces, normals):
    """Write triangulations."""
    with open(fname, 'w') as fid:
        fid.write("- %d\n" % len(points))
        for p, n in zip(points, normals):
            fid.write("%f %f %f" % tuple(p))
            fid.write(" %f %f %f\n" % tuple(n))
        fid.write("- %d %d %d\n" % ((len(faces),) * 3))
        [fid.write("%d %d %d\n" % tuple(p)) for p in faces]


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


def write_dipoles(fname, src, mri_trans, head_trans):
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
    np.savetxt(fname, np.c_[pos, ori])


# Write conductivities
COND_TEMPLATE = """# Properties Description 1.0 (Conductivities)

Air         0.0
Scalp       {2}
Brain       {0}
Skull       {1}
"""


def write_cond(fname, conductivity):
    """Write conductivities to .cond file."""
    with open(fname, 'w') as fid:
        fid.write(COND_TEMPLATE.format(*conductivity))

# Write geometry
GEOM_TEMPLATE = """# Domain Description 1.1

Interfaces 3

Interface Skull: "{1}.tri"
Interface Cortex: "{2}.tri"
Interface Head: "{0}.tri"

Domains 4

Domain Scalp: Skull -Head
Domain Brain: -Cortex
Domain Air: Head
Domain Skull: Cortex -Skull
"""


def write_geom(fname, surfs):
    """Write .geom file."""
    with open(fname, 'w') as fid:
        fid.write(GEOM_TEMPLATE.format(*surfs))


def write_eeg_locations(fname, info, head_trans):
    """Write channel location."""
    eeg_picks = mne.pick_types(info, meg=False, eeg=True, ref_meg=False)
    eeg_loc = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
    eeg_loc = apply_trans(head_trans, eeg_loc)

    np.savetxt(fname, eeg_loc)
    ch_names = [info['chs'][k]['ch_name'] for k in eeg_picks]
    return ch_names


def _make_forward(eeg_leadfield, ch_names, info, src, trans_fname):
    fwd = eeg_leadfield.array().astype(np.float32).T
    fwd = _to_forward_dict(fwd, ch_names)
    picks = mne.pick_channels(info['ch_names'], ch_names)
    fwd['info'] = mne.pick_info(info, picks)
    fwd['info']['mri_file'] = trans_fname
    fwd['info']['mri_id'] = fwd['info']['file_id']
    fwd['mri_head_t'] = invert_transform(mne.read_trans(trans_fname))
    fwd['info']['mri_head_t'] = fwd['mri_head_t']
    fwd['info']['meas_file'] = ""
    fwd['src'] = src
    fwd['surf_ori'] = False
    return fwd


def _get_geom_files(bem_model, mri_trans, head_trans):
    surf_names = ['inner_skull', 'outer_skull', 'outer_skin'][::-1]
    for surf_name, surf in zip(surf_names, bem_model):
        surf = _convert_bem_surf(surf, mri_trans, head_trans)
        points, faces, normals = surf['rr'], surf['tris'], surf['nn']
        # mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
        #                      faces, colormap='RdBu', opacity=0.5)
        write_tri('%s.tri' % surf_name, points, swap_faces(faces), normals)

    geom_fname = 'model.geom'
    cond_fname = 'model.cond'

    conductivity = [s['sigma'] for s in bem_model]

    write_geom(geom_fname, surf_names)
    write_cond(cond_fname, conductivity)

    coord_frame = 'head'
    trans = mne.read_trans(trans_fname)
    head_trans, meg_trans, mri_trans = _prepare_trans(info, trans, coord_frame)

    surf_names = ['inner_skull', 'outer_skull', 'outer_skin'][::-1]
    for surf_name, surf in zip(surf_names, bem_model):
        surf = _convert_bem_surf(surf, mri_trans, head_trans)
        points, faces, normals = surf['rr'], surf['tris'], surf['nn']
        # mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2],
        #                      faces, colormap='RdBu', opacity=0.5)
        write_tri('%s.tri' % surf_name, points, swap_faces(faces), normals)

    write_geom(geom_fname, surf_names)
    write_cond(cond_fname, conductivity)

    # OpenMEEG
    geom = om.Geometry(geom_fname, cond_fname)
    return geom


def _get_geom(bem_model, mri_trans, head_trans):
    surf_names = ['cortex', 'skull', 'scalp'][::-1]
    # meshes = []

    for surf_name, surf in zip(surf_names, bem_model):
        surf = _convert_bem_surf(surf, mri_trans, head_trans)
        points, faces, normals = surf['rr'], surf['tris'], surf['nn']
        write_tri('%s.tri' % surf_name, points, swap_faces(faces), normals)
        # mesh = om.Mesh('%s.tri' % surf_name)
        # mesh = om.Mesh(np.asfortranarray(points),
        #                np.asfortranarray(swap_faces(faces)).astype(np.int64))
        # mesh.update()
        # mesh.setName(surf_name)
        # meshes.append(mesh)

    # cortex, skull, scalp = meshes[::-1]

    cortex = om.Mesh("cortex.tri")
    cortex.setName("cortex")
    skull = om.Mesh("skull.tri")
    skull.setName("skull")
    scalp = om.Mesh("scalp.tri")
    scalp.setName("scalp")

    # cortex = om.Mesh("cortex.1.tri")
    # cortex.setName("cortex")
    # skull = om.Mesh("skull.1.tri")
    # skull.setName("skull")
    # scalp = om.Mesh("scalp.1.tri")
    # scalp.setName("scalp")

    # geom_fname = 'model.geom'
    # cond_fname = 'model.cond'

    conductivity = [s['sigma'] for s in bem_model]

    interfaces = {
        "interface1": [(cortex, om.OrientedMesh.Normal)],
        "interface2": [(skull, om.OrientedMesh.Normal)],
        "interface3": [(scalp, om.OrientedMesh.Normal)]
    }

    domains = {
        "Scalp": ([('interface2', om.SimpleDomain.Outside),
                   ('interface3', om.SimpleDomain.Inside)], conductivity[0]),
        "Brain": ([('interface1', om.SimpleDomain.Inside)], conductivity[2]),
        "Air": ([('interface3', om.SimpleDomain.Outside)], 0.0),
        "Skull": ([('interface2', om.SimpleDomain.Inside),
                   ('interface1', om.SimpleDomain.Outside)], conductivity[1])
    }

    geom = om.make_geometry(interfaces, domains)
    return geom


def _get_sensors_files(info, head_trans):
    electrodes_fname = 'electrodes.txt'
    ch_names = write_eeg_locations(electrodes_fname, info, head_trans)
    eeg_electrodes = om.Sensors(electrodes_fname)
    return eeg_electrodes, ch_names


def _get_sensors(info, head_trans):
    eeg_picks = mne.pick_types(info, meg=False, eeg=True, ref_meg=False)
    eeg_loc = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
    eeg_loc = np.asfortranarray(apply_trans(head_trans, eeg_loc))
    ch_names = [info['chs'][k]['ch_name'] for k in eeg_picks]
    orientations = np.ones((len(ch_names), 3), order='F')
    weights = np.ones(len(ch_names))
    radii = np.ones(len(ch_names))

    return om.Sensors(ch_names, eeg_loc, orientations, weights, radii),\
        ch_names


def _get_dipoles_files(src, mri_trans, head_trans):
    dipoles_fname = 'dipoles.txt'
    write_dipoles(dipoles_fname, src, mri_trans, head_trans)
    dipoles = om.Matrix(dipoles_fname)
    return dipoles


def _get_dipoles(src, mri_trans, head_trans):
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
    return om.Matrix(np.asfortranarray(np.c_[pos, ori]))


def make_forward_solution(info, trans_fname, src, bem_model, meg=True,
                          eeg=True, mindist=0.0, ignore_ref=False, n_jobs=1,
                          verbose=None):
    assert not meg  # XXX for now

    coord_frame = 'head'
    trans = mne.read_trans(trans_fname)
    head_trans, meg_trans, mri_trans = _prepare_trans(info, trans, coord_frame)

    dipoles = _get_dipoles(src, mri_trans, head_trans)
    eeg_electrodes, ch_names = _get_sensors(info, head_trans)

    geom = _get_geom_files(bem_model, mri_trans, head_trans)
    assert geom.is_nested()
    assert geom.selfCheck()

    # OpenMEEG
    gauss_order = 3
    use_adaptive_integration = True
    # dipole_in_cortex = True

    hm = om.HeadMat(geom, gauss_order)
    hm.invert()
    hminv = hm
    dsm = om.DipSourceMat(geom, dipoles, gauss_order,
                          use_adaptive_integration, "Brain")

    # For EEG
    eeg_picks = mne.pick_types(info, meg=False, eeg=True)
    # meg_picks = mne.pick_types(info, meg=True, eeg=False)
    # seeg_picks = mne.pick_types(info, meg=False, seeg=True)

    if eeg and len(eeg_picks) > 0:
        h2em = om.Head2EEGMat(geom, eeg_electrodes)
        eeg_leadfield = om.GainEEG(hminv, dsm, h2em)
        eegfwd = _make_forward(eeg_leadfield, ch_names, info, src, trans_fname)

    # if meg and len(meg_picks) > 0:
    #     h2em = om.Head2EEGMat(geom, eeg_electrodes)
    #     eeg_leadfield = om.GainEEG(hminv, dsm, h2em)
    #     megfwd = _make_forward(eeg_leadfield, ch_names, info, src, trans_fname)

    # # merge forwards
    # fwd = _merge_meg_eeg_fwds(_to_forward_dict(megfwd, megnames),
    #                           _to_forward_dict(eegfwd, eegnames),
    #                           verbose=False)

    return eegfwd


if __name__ == '__main__':
    from mne.datasets import sample
    data_path = sample.data_path()

    # the raw file containing the channel location + types
    raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
    # The paths to Freesurfer reconstructions
    subjects_dir = data_path + '/subjects'
    subject = 'sample'
    # The transformation file obtained by coregistration
    trans_fname = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

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
