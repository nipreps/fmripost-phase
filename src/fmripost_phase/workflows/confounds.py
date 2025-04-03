# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Calculate BOLD confounds."""


def init_bold_confs_wf(
    mem_gb: float,
    metadata: dict,
    regressors_all_comps: bool,
    name: str = 'bold_confs_wf',
):
    """Build a workflow to generate and write out confounding signals.

    This workflow calculates confounds for a BOLD series, and aggregates them
    into a :abbr:`TSV (tab-separated value)` file, for use as nuisance
    regressors in a :abbr:`GLM (general linear model)`.
    The following confounds are calculated, with column headings in parentheses:

    #. HighCor (``h_comp_cor_XX``)

    Prior to estimating hCompCor, non-steady-state volumes are
    censored and high-pass filtered using a :abbr:`DCT (discrete cosine transform)` basis.
    The cosine basis, as well as one regressor per censored volume, are included
    for convenience.

    XXX: What about tissue time series from the jolt and/or jump files?
    """
    from bids.utils import listify
    from fmriprep.interfaces.confounds import FilterDropped
    from fmriprep.utils.bids import dismiss_echo
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from nireports.interfaces.nuisance import CompCorVariancePlot
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.images import SignalExtraction
    from niworkflows.interfaces.patches import RobustTCompCor as TCompCor
    from niworkflows.interfaces.reportlets.masks import ROIsPlot
    from niworkflows.interfaces.utility import TSV2JSON

    from fmripost_phase.config import DEFAULT_MEMORY_MIN_GB
    from fmripost_phase.interfaces.bids import DerivativesDataSink
    from fmripost_phase.interfaces.confounds import GatherConfounds
    from fmripost_phase.utils.utils import clean_datasinks

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'phase',
                'bold_mask',
                'skip_vols',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'confounds_file',
                'confounds_metadata',
                'highcor_mask',
            ],
        ),
        name='outputnode',
    )

    highcor = pe.Node(
        TCompCor(
            components_file='highcor.tsv',
            header_prefix='h_comp_cor_',
            pre_filter='cosine',
            save_pre_filter=True,
            save_metadata=True,
            percentile_threshold=0.02,
            failure_mode='NaN',
        ),
        name='highcor',
        mem_gb=mem_gb,
    )

    # Set number of components
    if regressors_all_comps:
        highcor.inputs.num_components = 'all'
    else:
        highcor.inputs.variance_threshold = 0.5

    # Set TR if present
    if 'RepetitionTime' in metadata:
        highcor.inputs.repetition_time = metadata['RepetitionTime']

    # Global and segment regressors
    signals_class_labels = ['highcor']
    signals = pe.Node(
        SignalExtraction(class_labels=signals_class_labels),
        name='signals',
        mem_gb=mem_gb,
    )
    concat = pe.Node(GatherConfounds(), name='concat', mem_gb=0.01, run_without_submitting=True)

    # CompCor metadata
    hcc_metadata_filter = pe.Node(FilterDropped(), name='hcc_metadata_filter')
    hcc_metadata_fmt = pe.Node(
        TSV2JSON(
            index_column='component',
            drop_columns=['mask'],
            output=None,
            additional_metadata={'Method': 'HighCor'},
            enforce_case=True,
        ),
        name='hcc_metadata_fmt',
    )

    # Generate reportlet (ROIs)
    rois_plot = pe.Node(
        ROIsPlot(colors=['b'], generate_report=True),
        name='rois_plot',
        mem_gb=mem_gb,
    )
    ds_report_bold_rois = pe.Node(
        DerivativesDataSink(desc='rois', dismiss_entities=dismiss_echo()),
        name='ds_report_bold_rois',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # Generate reportlet (CompCor)
    compcor_plot = pe.Node(
        CompCorVariancePlot(
            variance_thresholds=(0.5, 0.7, 0.9),
            metadata_sources=['HighCor'],
        ),
        name='compcor_plot',
    )
    ds_report_compcor = pe.Node(
        DerivativesDataSink(desc='compcorvar', dismiss_entities=dismiss_echo()),
        name='ds_report_compcor',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (inputnode, highcor, [
            ('phase', 'realigned_file'),
            ('skip_vols', 'ignore_initial_volumes'),
            ('bold_mask', 'mask_files'),
        ]),
        (highcor, signals, [('high_variance_masks', 'label_files')]),
        (signals, concat, [('out_file', 'signals')]),
        (highcor, concat, [
            ('components_file', 'highcor'),
            ('pre_filter_file', 'cos_basis'),
        ]),
        (highcor, hcc_metadata_filter, [('metadata_file', 'in_file')]),
        (hcc_metadata_filter, hcc_metadata_fmt, [('out_file', 'in_file')]),
        (hcc_metadata_fmt, outputnode, [('output', 'confounds_metadata')]),
        (highcor, outputnode, [('high_variance_masks', 'highcor_mask')]),
        (highcor, rois_plot, [('high_variance_masks', 'in_rois')]),
        (rois_plot, ds_report_bold_rois, [('out_report', 'in_file')]),
        (highcor, compcor_plot, [(('metadata_file', listify), 'metadata_files')]),
        (compcor_plot, ds_report_compcor, [('out_file', 'in_file')]),
    ])  # fmt:skip

    return clean_datasinks(workflow)


def init_carpetplot_wf(
    mem_gb: float,
    metadata: dict,
    cifti_output: bool,
    name: str = 'bold_carpet_wf',
):
    """Build a workflow to generate *carpet* plots.

    Resamples the MNI parcellation
    (ad-hoc parcellation derived from the Harvard-Oxford template and others).

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB - please note that this size
        should be calculated after resamplings that may extend
        the FoV
    metadata : :obj:`dict`
        BIDS metadata for BOLD file
    name : :obj:`str`
        Name of workflow (default: ``bold_carpet_wf``)

    Inputs
    ------
    bold
        BOLD image, in MNI152NLin6Asym space + 2mm resolution.
    bold_mask
        BOLD series mask in same space as ``bold``.
    confounds_file
        TSV of all aggregated confounds
    boldref2anat_xfm
        Transform from boldref to anat space
    std2anat_xfm
        Transform from standard space to anat space
    cifti_bold
        BOLD image in CIFTI format, to be used in place of volumetric BOLD
    crown_mask
        Mask of brain edge voxels. Dropped.
    acompcor_mask
        Mask of deep WM+CSF. Dropped.
    dummy_scans
        Number of nonsteady states to be dropped at the beginning of the timeseries.
    desc
        Description of the carpet plot.

    Outputs
    -------
    out_carpetplot
        Path of the generated SVG file
    """
    from fmriprep.interfaces.confounds import FMRISummary
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from templateflow.api import get as get_template

    from fmripost_phase.config import DEFAULT_MEMORY_MIN_GB
    from fmripost_phase.interfaces.bids import DerivativesDataSink

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'bold',
                'bold_mask',
                'confounds_file',
                'boldref2anat_xfm',
                'std2anat_xfm',
                'cifti_bold',
                'dummy_scans',
                'desc',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(niu.IdentityInterface(fields=['out_carpetplot']), name='outputnode')

    # Carpetplot and confounds plot
    conf_plot = pe.Node(
        FMRISummary(
            tr=metadata['RepetitionTime'],
            confounds_list=[
                ('trans_x', 'mm', 'x'),
                ('trans_y', 'mm', 'y'),
                ('trans_z', 'mm', 'z'),
                ('rot_x', 'deg', 'pitch'),
                ('rot_y', 'deg', 'roll'),
                ('rot_z', 'deg', 'yaw'),
                ('framewise_displacement', 'mm', 'FD'),
            ],
        ),
        name='conf_plot',
        mem_gb=mem_gb,
    )
    ds_report_bold_conf = pe.Node(
        DerivativesDataSink(
            datatype='figures',
            extension='svg',
            dismiss_entities=('echo', 'den', 'res'),
        ),
        name='ds_report_bold_conf',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    parcels = pe.Node(niu.Function(function=_carpet_parcellation), name='parcels')
    parcels.inputs.nifti = not cifti_output

    # Warp segmentation into MNI152NLin6Asym space
    resample_parc = pe.Node(
        ApplyTransforms(
            dimension=3,
            input_image=str(
                get_template(
                    'MNI152NLin2009cAsym',
                    resolution=1,
                    desc='carpet',
                    suffix='dseg',
                    extension=['.nii', '.nii.gz'],
                )
            ),
            invert_transform_flags=[True, False],
            interpolation='MultiLabel',
            args='-u int',
        ),
        name='resample_parc',
    )

    workflow = Workflow(name=name)
    # List transforms
    mrg_xfms = pe.Node(niu.Merge(2), name='mrg_xfms')
    if cifti_output:
        workflow.connect(inputnode, 'cifti_bold', conf_plot, 'in_cifti')

    workflow.connect([
        (inputnode, mrg_xfms, [
            ('boldref2anat_xfm', 'in1'),
            ('std2anat_xfm', 'in2'),
        ]),
        (inputnode, resample_parc, [('bold_mask', 'reference_image')]),
        (mrg_xfms, resample_parc, [('out', 'transforms')]),
        (inputnode, conf_plot, [
            ('bold', 'in_nifti'),
            ('confounds_file', 'confounds_file'),
            ('dummy_scans', 'drop_trs'),
        ]),
        (resample_parc, parcels, [('output_image', 'segmentation')]),
        (parcels, conf_plot, [('out', 'in_segm')]),
        (inputnode, ds_report_bold_conf, [('desc', 'desc')]),
        (conf_plot, ds_report_bold_conf, [('out_file', 'in_file')]),
        (conf_plot, outputnode, [('out_file', 'out_carpetplot')]),
    ])  # fmt:skip
    return workflow


def _carpet_parcellation(segmentation, nifti=False):
    """Generate the union of two masks."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)

    lut = np.zeros((256,), dtype='uint8')
    lut[100:201] = 1 if nifti else 0  # Ctx GM
    lut[30:99] = 2 if nifti else 0  # dGM
    lut[1:11] = 3 if nifti else 1  # WM+CSF
    lut[255] = 5 if nifti else 0  # Cerebellum
    # Apply lookup table
    seg = lut[np.uint16(img.dataobj)]
    # seg[np.bool_(nb.load(crown_mask).dataobj)] = 6 if nifti else 2
    # Separate deep from shallow WM+CSF
    # seg[np.bool_(nb.load(acompcor_mask).dataobj)] = 4 if nifti else 1

    outimg = img.__class__(seg.astype('uint8'), img.affine, img.header)
    outimg.set_data_dtype('uint8')
    out_file = Path('segments.nii.gz').absolute()
    outimg.to_filename(out_file)
    return str(out_file)
