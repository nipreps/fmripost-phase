# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
"""
fMRIPost phase workflows
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_fmripost_phase_wf
.. autofunction:: init_single_subject_wf

"""

import os
import sys
from collections import defaultdict
from copy import deepcopy

import yaml
from nipype.pipeline import engine as pe
from packaging.version import Version

from fmripost_phase import config
from fmripost_phase.utils.utils import _get_wf_name, clean_datasinks, update_dict


def init_fmripost_phase_wf():
    """Build *fMRIPost-Phase*'s pipeline.

    This workflow organizes the execution of fMRIPost-Phase,
    with a sub-workflow for each subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_phase.workflows.tests import mock_config
            from fmripost_phase.workflows.base import init_fmripost_phase_wf

            with mock_config():
                wf = init_fmripost_phase_wf()

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    ver = Version(config.environment.version)

    fmripost_phase_wf = Workflow(name=f'fmripost_phase_{ver.major}_{ver.minor}_wf')
    fmripost_phase_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config['execution']['crashdump_dir'] = str(
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        fmripost_phase_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.output_dir / f'sub-{subject_id}' / 'log' / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / 'fmripost_phase.toml')

    return fmripost_phase_wf


def init_single_subject_wf(subject_id: str):
    """Organize the postprocessing pipeline for a single subject.

    It collects and reports information about the subject,
    and prepares sub-workflows to postprocess each BOLD series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from fmripost_phase.workflows.tests import mock_config
            from fmripost_phase.workflows.base import init_single_subject_wf

            with mock_config():
                wf = init_single_subject_wf('01')

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label for this single-subject workflow.

    Notes
    -----
    1.  Load fMRIPost-Phase config file.
    2.  Collect fMRIPrep derivatives.
        -   BOLD file in native space.
        -   Two main possibilities:
            1.  bids_dir is a raw BIDS dataset and preprocessing derivatives
                are provided through ``--derivatives``.
                In this scenario, we only need minimal derivatives.
            2.  bids_dir is a derivatives dataset and we need to collect compliant
                derivatives to get the data into the right space.
    3.  Loop over runs.
    4.  Collect each run's associated files.
        -   Transform(s) to MNI152NLin6Asym
        -   Confounds file
    5.  Run phase processing steps.
    8.  Warp BOLD to requested output spaces and denoise with ICA-Phase.

    """
    from bids.utils import listify
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.bids import BIDSInfo
    from niworkflows.interfaces.nilearn import NILEARN_VERSION

    from fmripost_phase.interfaces.bids import DerivativesDataSink
    from fmripost_phase.interfaces.reportlets import AboutSummary, SubjectSummary
    from fmripost_phase.utils.bids import collect_derivatives

    spaces = config.workflow.spaces

    workflow = Workflow(name=f'sub_{subject_id}_wf')
    workflow.__desc__ = f"""
Results included in this manuscript come from postprocessing
performed using *fMRIPost-Phase* {config.environment.version} (@ica_phase),
which is based on *Nipype* {config.environment.nipype_version}
(@nipype1; @nipype2; RRID:SCR_002502).

"""
    workflow.__postdesc__ = f"""

Many internal operations of *fMRIPost-Phase* use
*Nilearn* {NILEARN_VERSION} [@nilearn, RRID:SCR_001362].
For more details of the pipeline, see [the section corresponding
to workflows in *fMRIPost-Phase*'s documentation]\
(https://fmripost_phase.readthedocs.io/en/latest/workflows.html \
"FMRIPrep's documentation").


### Copyright Waiver

The above boilerplate text was automatically generated by fMRIPost-Phase
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

### References

"""

    if config.execution.derivatives:
        # Raw dataset + derivatives dataset
        config.loggers.workflow.info('Raw+derivatives workflow mode enabled')
        subject_data = collect_derivatives(
            raw_dataset=config.execution.layout,
            derivatives_dataset=None,
            entities=config.execution.bids_filters,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        subject_data['bold'] = listify(subject_data['magnitude_raw'])
    else:
        # Derivatives dataset only
        config.loggers.workflow.info('Derivatives-only workflow mode enabled')
        subject_data = collect_derivatives(
            raw_dataset=None,
            derivatives_dataset=config.execution.layout,
            entities=config.execution.bids_filters,
            fieldmap_id=None,
            allow_multiple=True,
            spaces=None,
        )
        # Patch standard-space BOLD files into 'bold' key
        subject_data['bold'] = listify(subject_data['bold_mni152nlin6asym'])

    # Make sure we always go through these two checks
    if not subject_data['bold']:
        task_id = config.execution.task_id
        raise RuntimeError(
            f"No BOLD images found for participant {subject_id} and "
            f"task {task_id if task_id else '<all>'}. "
            "All workflows require BOLD images. "
            f"Please check your BIDS filters: {config.execution.bids_filters}."
        )

    bids_info = pe.Node(
        BIDSInfo(
            bids_dir=config.execution.bids_dir,
            bids_validate=False,
            in_file=subject_data['bold'][0],
        ),
        name='bids_info',
    )

    summary = pe.Node(
        SubjectSummary(
            bold=subject_data['bold'],
            std_spaces=spaces.get_spaces(nonstandard=False),
            nstd_spaces=spaces.get_spaces(standard=False),
        ),
        name='summary',
        run_without_submitting=True,
    )
    workflow.connect([(bids_info, summary, [('subject', 'subject_id')])])

    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)),
        name='about',
        run_without_submitting=True,
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='summary',
            datatype='figures',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
    )
    workflow.connect([(summary, ds_report_summary, [('out_report', 'in_file')])])

    ds_report_about = pe.Node(
        DerivativesDataSink(
            source_file=subject_data['bold'][0],
            base_directory=config.execution.output_dir,
            desc='about',
            datatype='figures',
        ),
        name='ds_report_about',
        run_without_submitting=True,
    )
    workflow.connect([(about, ds_report_about, [('out_report', 'in_file')])])

    # Append the functional section to the existing anatomical excerpt
    # That way we do not need to stream down the number of bold datasets
    func_pre_desc = f"""
Functional data postprocessing

: For each of the {len(subject_data['bold'])} BOLD runs found per subject
(across all tasks and sessions), the following postprocessing was performed.
"""
    workflow.__desc__ += func_pre_desc

    for bold_file in subject_data['bold']:
        single_run_wf = init_single_run_wf(bold_file)
        workflow.add_nodes([single_run_wf])

    return clean_datasinks(workflow)


def init_single_run_wf(bold_file):
    """Set up a single-run workflow for fMRIPost-Phase."""
    from fmriprep.utils.bids import dismiss_echo
    from fmriprep.utils.misc import estimate_bold_mem_usage
    from fmriprep.workflows.bold.apply import init_bold_volumetric_resample_wf
    from fmriprep.workflows.bold.stc import init_bold_stc_wf
    from nipype.interfaces import utility as niu
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.header import ValidateImage

    from fmripost_phase.interfaces.bids import DerivativesDataSink
    from fmripost_phase.interfaces.complex import Phase2Radians
    from fmripost_phase.interfaces.laynii import LayNiiPhaseJolt
    from fmripost_phase.utils.bids import collect_derivatives, extract_entities
    from fmripost_phase.workflows.confounds import init_bold_confs_wf
    from fmripost_phase.workflows.regression import init_phase_regression_wf

    spaces = config.workflow.spaces
    omp_nthreads = config.nipype.omp_nthreads

    workflow = Workflow(name=_get_wf_name(bold_file, 'single_run'))
    workflow.__desc__ = ''

    bold_metadata = config.execution.layout.get_metadata(bold_file)
    mem_gb = estimate_bold_mem_usage(bold_file)[1]

    entities = extract_entities(bold_file)

    functional_cache = defaultdict(list, {})
    # Collect native-space derivatives and transforms
    functional_cache = collect_derivatives(
        raw_dataset=config.execution.layout,
        derivatives_dataset=None,
        entities=entities,
        fieldmap_id=None,
        allow_multiple=False,
        spaces=None,
    )
    for deriv_dir in config.execution.derivatives.values():
        functional_cache = update_dict(
            functional_cache,
            collect_derivatives(
                raw_dataset=None,
                derivatives_dataset=deriv_dir,
                entities=entities,
                fieldmap_id=None,
                allow_multiple=False,
                spaces=spaces,
            ),
        )

    if not functional_cache['confounds']:
        if config.workflow.dummy_scans is None:
            raise ValueError(
                'No confounds detected. '
                'Automatical dummy scan detection cannot be performed. '
                'Please set the `--dummy-scans` flag explicitly.'
            )

        # TODO: Calculate motion parameters from motion correction transform
        raise NotImplementedError('Motion parameters cannot be extracted from transforms yet.')

    config.loggers.workflow.info(
        (
            f'Collected run data for {os.path.basename(bold_file)}:\n'
            f'{yaml.dump(functional_cache, default_flow_style=False, indent=4)}'
        ),
    )

    if config.workflow.dummy_scans is not None:
        skip_vols = config.workflow.dummy_scans
    else:
        if not functional_cache['confounds']:
            raise ValueError(
                'No confounds detected. '
                'Automatical dummy scan detection cannot be performed. '
                'Please set the `--dummy-scans` flag explicitly.'
            )
        skip_vols = get_nss(functional_cache['confounds'])

    validate_bold = pe.Node(
        ValidateImage(in_file=functional_cache['magnitude_raw']),
        name='validate_bold',
    )

    phase_buffer = pe.Node(
        niu.IdentityInterface(fields=['phase', 'phase_norf']),
        name='phase_buffer',
    )
    has_norf = (
        ('norf' not in config.workflow.ignore) and
        ('phase_norf' in functional_cache) and
        ('magnitude_norf' in functional_cache) and
        config.workflow.thermal_denoise_method
    )
    if has_norf:
        from fmripost_phase.interfaces.complex import ConcatenateNoise, SplitNoise

        # Concatenate phase and noRF data before rescaling
        concatenate_phase = pe.Node(
            ConcatenateNoise(
                in_file=functional_cache['phase_raw'],
                noise_file=functional_cache['phase_norf'],
            ),
            name='concatenate_phase',
        )

        # Rescale phase data to radians
        phase_to_radians = pe.Node(
            Phase2Radians(),
            name='phase_to_radians',
        )
        workflow.connect([(concatenate_phase, phase_to_radians, [('out', 'in_file')])])

        # Split rescaled phase data
        split_phase = pe.Node(
            SplitNoise(),
            name='split_phase',
        )
        workflow.connect([
            (concatenate_phase, split_phase, [('n_noise_volumes', 'n_noise_volumes')]),
            (phase_to_radians, split_phase, [('out_file', 'in_file')]),
            (split_phase, phase_buffer, [
                ('out_file', 'phase'),
                ('noise_file', 'phase_norf'),
            ]),
        ])  # fmt:skip
    else:
        # Rescale phase data to radians
        phase_to_radians = pe.Node(
            Phase2Radians(),
            name='phase_to_radians',
        )
        phase_to_radians.inputs.in_file = functional_cache['phase_raw']
        workflow.connect([(phase_to_radians, phase_buffer, [('out_file', 'phase')])])

    denoise_buffer = pe.Node(
        niu.IdentityInterface(fields=['magnitude', 'phase']),
        name='denoise_buffer',
    )
    if config.workflow.thermal_denoise_method:
        # Run LLR denoising on the magnitude and phase data
        denoise_wf = pe.Node(
            niu.IdentityInterface(fields=['magnitude', 'phase', 'magnitude_norf', 'phase_norf']),
            name='denoise_wf',
        )
        if has_norf:
            validate_norf = pe.Node(
                ValidateImage(in_file=functional_cache['magnitude_norf']),
                name='validate_norf',
            )
            workflow.connect([
                (validate_norf, denoise_wf, [('out_file', 'inputnode.magnitude_norf')]),
                (phase_buffer, denoise_wf, [('phase_norf', 'inputnode.phase_norf')]),
            ])  # fmt:skip

        workflow.connect([
            (validate_bold, denoise_wf, [('out_file', 'inputnode.magnitude')]),
            (phase_buffer, denoise_wf, [('phase', 'inputnode.phase')]),
            (denoise_wf, denoise_buffer, [
                ('outputnode.magnitude', 'magnitude'),
                ('outputnode.phase', 'phase'),
            ]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (validate_bold, denoise_buffer, [('out_file', 'magnitude')]),
            (phase_buffer, denoise_buffer, [('phase', 'phase')]),
        ])  # fmt:skip

    # Warp magnitude data to boldref space
    stc_buffer = pe.Node(
        niu.IdentityInterface(fields=['bold_file']),
        name='stc_buffer',
    )
    run_stc = ('SliceTiming' in bold_metadata) and 'slicetiming' not in config.workflow.ignore
    if run_stc:
        bold_stc_wf = init_bold_stc_wf(
            mem_gb=mem_gb,
            metadata=bold_metadata,
            name='resample_stc_wf',
        )
        bold_stc_wf.inputs.inputnode.skip_vols = skip_vols
        workflow.connect([
            (denoise_buffer, bold_stc_wf, [('magnitude', 'inputnode.bold_file')]),
            (bold_stc_wf, stc_buffer, [('outputnode.stc_file', 'bold_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([(denoise_buffer, stc_buffer, [('magnitude', 'bold_file')])])

    mag_boldref_wf = init_bold_volumetric_resample_wf(
        metadata=bold_metadata,
        fieldmap_id=None,  # XXX: Ignoring the field map for now
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        jacobian='fmap-jacobian' not in config.workflow.ignore,
        name='mag_boldref_wf',
    )
    mag_boldref_wf.inputs.inputnode.motion_xfm = functional_cache['hmc']
    mag_boldref_wf.inputs.inputnode.boldref2fmap_xfm = functional_cache['boldref2fmap']
    mag_boldref_wf.inputs.inputnode.bold_ref_file = functional_cache['bold_mask_native']

    workflow.connect([
        (stc_buffer, mag_boldref_wf, [('bold_file', 'inputnode.bold_file')]),
        # XXX: Ignoring the field map for now
        # (inputnode, mag_boldref_wf, [
        #     ('fmap_ref', 'inputnode.fmap_ref'),
        #     ('fmap_coeff', 'inputnode.fmap_coeff'),
        #     ('fmap_id', 'inputnode.fmap_id'),
        # ]),
    ])  # fmt:skip

    # Minimally process phase data
    # Remove non-steady-state volumes
    remove_phase_nss = pe.Node(
        niu.IdentityInterface(fields=['phase_file', 'skip_vols']),
        name='remove_phase_nss',
    )
    remove_phase_nss.inputs.skip_vols = skip_vols
    workflow.connect([(denoise_buffer, remove_phase_nss, [('phase', 'phase_file')])])

    # Unwrap with warpkit
    unwrap_wf = pe.Node(
        niu.IdentityInterface(fields=['phase_file']),
        name='unwrap_wf',
    )
    workflow.connect([(remove_phase_nss, unwrap_wf, [('phase_file', 'phase_file')])])

    # Warp to boldref space (motion correction + distortion correction[?] + fmap-to-boldref)
    phase_boldref_wf = init_bold_volumetric_resample_wf(
        metadata=bold_metadata,
        fieldmap_id=None,  # XXX: Ignoring the field map for now
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        jacobian='fmap-jacobian' not in config.workflow.ignore,
        name='mag_boldref_wf',
    )
    phase_boldref_wf.inputs.inputnode.motion_xfm = functional_cache['hmc']
    phase_boldref_wf.inputs.inputnode.boldref2fmap_xfm = functional_cache['boldref2fmap']
    phase_boldref_wf.inputs.inputnode.bold_ref_file = functional_cache['bold_mask_native']
    workflow.connect([(unwrap_wf, phase_boldref_wf, [('out_file', 'inputnode.bold_file')])])

    if config.workflow.retroicor:
        # Run RETROICOR on the magnitude and phase data
        # After rescaling + unwrapping
        # TODO: Load physio data
        raise NotImplementedError('RETROICOR is not yet implemented.')

    if config.workflow.regression_method:
        # Now denoise the BOLD data using phase regression
        phase_regression_wf = init_phase_regression_wf(bold_file=bold_file, metadata=bold_metadata)
        phase_regression_wf.inputs.inputnode.skip_vols = skip_vols
        phase_regression_wf.inputs.inputnode.bold_mask = functional_cache['bold_mask_native']

        workflow.connect([
            (phase_boldref_wf, phase_regression_wf, [
                ('outputnode.bold_file', 'inputnode.phase_file'),
            ]),
            (mag_boldref_wf, phase_regression_wf, [
                ('outputnode.bold_file', 'inputnode.bold_file'),
            ]),
        ])  # fmt:skip

    # Calculate phase jolt and/or jump files
    if config.workflow.jolt:
        # Calculate phase jolt
        calc_jolt = pe.Node(
            LayNiiPhaseJolt(phase_jump=False),
            name='calc_jolt',
        )
        workflow.connect([(phase_boldref_wf, calc_jolt, [('outputnode.out_file', 'in_file')])])

        ds_jolt = pe.Node(
            DerivativesDataSink(
                source_file=bold_file,
                desc='jolt',
            ),
            name='ds_jolt',
        )
        workflow.connect([(calc_jolt, ds_jolt, [('out_file', 'in_file')])])

    if config.workflow.jump:
        # Calculate phase jump
        calc_jump = pe.Node(
            LayNiiPhaseJolt(phase_jump=True),
            name='calc_jump',
        )
        workflow.connect([(phase_boldref_wf, calc_jump, [('outputnode.out_file', 'in_file')])])

        ds_jump = pe.Node(
            DerivativesDataSink(
                source_file=bold_file,
                desc='jump',
            ),
            name='ds_jump',
        )
        workflow.connect([(calc_jump, ds_jump, [('out_file', 'in_file')])])

    if config.workflow.gift_dimensionality != 0:
        # Run GIFT ICA
        raise NotImplementedError('GIFT ICA is not yet implemented.')

    # Compute confounds, including HighCor
    bold_confounds_wf = init_bold_confs_wf(
        mem_gb=mem_gb['largemem'],
        metadata=bold_metadata,
        regressors_all_comps=config.workflow.regressors_all_comps,
        name='bold_confounds_wf',
    )
    bold_confounds_wf.inputs.inputnode.skip_vols = skip_vols
    bold_confounds_wf.inputs.inputnode.bold_mask = functional_cache['bold_mask_native']
    workflow.connect([
        (phase_boldref_wf, bold_confounds_wf, [('outputnode.bold_file', 'inputnode.phase')]),
    ])  # fmt:skip

    ds_confounds = pe.Node(
        DerivativesDataSink(
            desc='confounds',
            suffix='timeseries',
            dismiss_entities=dismiss_echo(),
        ),
        name='ds_confounds',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([
        (bold_confounds_wf, ds_confounds, [
            ('outputnode.confounds_file', 'in_file'),
            ('outputnode.confounds_metadata', 'meta_dict'),
        ]),
    ])  # fmt:skip

    # Fill-in datasinks seen so far
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).inputs.base_directory = config.execution.output_dir
            workflow.get_node(node).inputs.source_file = bold_file

    return workflow


def _prefix(subid):
    return subid if subid.startswith('sub-') else f'sub-{subid}'


def get_nss(confounds_file):
    """Get number of non-steady state volumes."""
    import numpy as np
    import pandas as pd

    df = pd.read_table(confounds_file)

    nss_cols = [c for c in df.columns if c.startswith('non_steady_state_outlier')]

    dummy_scans = 0
    if nss_cols:
        initial_volumes_df = df[nss_cols]
        dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
        dummy_scans = np.where(dummy_scans)[0]

        # reasonably assumes all NSS volumes are contiguous
        dummy_scans = int(dummy_scans[-1] + 1)

    return dummy_scans
