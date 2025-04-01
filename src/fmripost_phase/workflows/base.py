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
from nipype.interfaces import utility as niu
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
    from niworkflows.interfaces.utility import KeySelect

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
            f'No BOLD images found for participant {subject_id} and '
            f'task {task_id if task_id else "<all>"}. '
            'All workflows require BOLD images. '
            f'Please check your BIDS filters: {config.execution.bids_filters}.'
        )

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['std2anat_xfm', 'template']),
        name='inputnode',
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

    if 'MNI152NLin2009cAsym' in spaces.get_spaces():
        select_MNI2009c_xfm = pe.Node(
            KeySelect(fields=['std2anat_xfm'], key='MNI152NLin2009cAsym'),
            name='select_MNI2009c_xfm',
            run_without_submitting=True,
        )
        workflow.connect([
            (inputnode, select_MNI2009c_xfm, [
                ('std2anat_xfm', 'std2anat_xfm'),
                ('template', 'keys'),
            ]),
        ])  # fmt:skip

    for bold_file in subject_data['bold']:
        single_run_wf = init_single_run_wf(bold_file)
        workflow.connect([
            (inputnode, single_run_wf, [
                ('std2anat_xfm', 'inputnode.std2anat_xfm'),
                ('template', 'inputnode.template'),
            ]),
            (select_MNI2009c_xfm, single_run_wf, [
                ('mni2009c2anat_xfm', 'inputnode.mni2009c2anat_xfm'),
            ]),
        ])  # fmt:skip

    return clean_datasinks(workflow)


def init_single_run_wf(bold_file):
    """Set up a single-run workflow for fMRIPost-Phase."""
    from fmriprep.utils.bids import dismiss_echo
    from fmriprep.utils.misc import estimate_bold_mem_usage
    from fmriprep.workflows.bold.apply import init_bold_volumetric_resample_wf
    from fmriprep.workflows.bold.stc import init_bold_stc_wf
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.header import ValidateImage

    from fmripost_phase.interfaces.bids import DerivativesDataSink
    from fmripost_phase.interfaces.complex import Scale2Radians
    from fmripost_phase.interfaces.laynii import LayNiiPhaseJolt, LayNiiPhaseLaplacian
    from fmripost_phase.interfaces.misc import DictToJSON, RemoveNSS
    from fmripost_phase.interfaces.warpkit import ROMEOUnwrap, WarpkitUnwrap
    from fmripost_phase.utils.bids import collect_derivatives, extract_entities
    from fmripost_phase.workflows.confounds import init_bold_confs_wf, init_carpetplot_wf
    from fmripost_phase.workflows.regression import init_phase_regression_wf

    spaces = config.workflow.spaces
    nonstd_spaces = set(spaces.get_nonstandard())
    freesurfer_spaces = set(spaces.get_fs_spaces())
    omp_nthreads = config.nipype.omp_nthreads

    workflow = Workflow(name=_get_wf_name(bold_file, 'single_run'))
    workflow.__desc__ = ''

    bold_metadata = config.execution.layout.get_metadata(bold_file)
    mem_gb = estimate_bold_mem_usage(bold_file)[1]
    multiecho = isinstance(bold_file, list)  # XXX: This won't work

    entities = extract_entities(bold_file)

    # Attempt to extract the associated fmap ID
    fmapid = None
    all_fmapids = config.execution.layout.get_fmapids(
        subject=entities['subject'],
        session=entities.get('session', None),
    )
    if all_fmapids:
        fmap_file = config.execution.layout.get_nearest(
            bold_file,
            to=all_fmapids,
            suffix='xfm',
            extension='.txt',
            strict=False,
            **{'from': 'boldref'},
        )
        if fmap_file:
            fmapid = config.execution.layout.get_file(fmap_file).entities['to']

    functional_cache = defaultdict(list, {})
    # Collect native-space derivatives and transforms
    functional_cache = collect_derivatives(
        raw_dataset=config.execution.layout,
        derivatives_dataset=None,
        entities=entities,
        fieldmap_id=fmapid,
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
                fieldmap_id=fmapid,
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

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude_raw',
                'phase_raw',
                'magnitude_norf',
                'phase_norf',
                'hmc',
                'boldref2fmap',
                'bold_mask_native',
                'confounds',
                'mni2009c2anat_xfm',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.magnitude_raw = functional_cache['magnitude_raw']
    inputnode.inputs.phase_raw = functional_cache['phase_raw']
    inputnode.inputs.magnitude_norf = functional_cache['magnitude_norf']
    inputnode.inputs.phase_norf = functional_cache['phase_norf']
    inputnode.inputs.hmc = functional_cache['hmc']
    inputnode.inputs.boldref2fmap = functional_cache['boldref2fmap']
    inputnode.inputs.bold_mask_native = functional_cache['bold_mask_native']
    inputnode.inputs.confounds = functional_cache['confounds']
    # Field maps
    inputnode.inputs.fmap = functional_cache['fmap']

    validate_bold = pe.Node(
        ValidateImage(),
        name='validate_bold',
    )
    workflow.connect([(inputnode, validate_bold, [('magnitude_raw', 'in_file')])])

    phase_buffer = pe.Node(
        niu.IdentityInterface(fields=['phase', 'phase_norf']),
        name='phase_buffer',
    )
    has_norf = (
        ('norf' not in config.workflow.ignore)
        and ('phase_norf' in functional_cache)
        and ('magnitude_norf' in functional_cache)
        and config.workflow.thermal_denoise_method
    )
    if has_norf:
        from fmripost_phase.interfaces.complex import ConcatenateNoise, SplitNoise

        # Concatenate phase and noRF data before rescaling
        concatenate_phase = pe.Node(
            ConcatenateNoise(),
            name='concatenate_phase',
        )
        workflow.connect([
            (inputnode, concatenate_phase, [
                ('phase_raw', 'in_file'),
                ('phase_norf', 'noise_file'),
            ]),
        ])  # fmt:skip

        # Rescale phase data to radians (-pi to pi)
        # XXX: Why is this needed?
        phase_to_radians = pe.Node(
            Scale2Radians(scale='pi'),
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
        # Rescale phase data to radians (-pi to pi)
        phase_to_radians = pe.Node(
            Scale2Radians(scale='pi'),
            name='phase_to_radians',
        )
        workflow.connect([
            (inputnode, phase_to_radians, [('phase_raw', 'in_file')]),
            (phase_to_radians, phase_buffer, [('out_file', 'phase')]),
        ])  # fmt:skip

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
                ValidateImage(),
                name='validate_norf',
            )
            workflow.connect([
                (inputnode, validate_norf, [('magnitude_norf', 'in_file')]),
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

    # Apply slice timing correction, if requested
    stc_buffer = pe.Node(
        niu.IdentityInterface(fields=['bold_file']),
        name='stc_buffer',
    )
    run_stc = ('SliceTiming' in bold_metadata) and 'slicetiming' not in config.workflow.ignore
    if run_stc:
        mag_stc_wf = init_bold_stc_wf(
            mem_gb=mem_gb,
            metadata=bold_metadata,
            name='mag_stc_wf',
        )
        mag_stc_wf.inputs.inputnode.skip_vols = skip_vols
        workflow.connect([
            (denoise_buffer, mag_stc_wf, [('magnitude', 'inputnode.bold_file')]),
            (mag_stc_wf, stc_buffer, [('outputnode.stc_file', 'bold_file')]),
        ])  # fmt:skip
    else:
        workflow.connect([(denoise_buffer, stc_buffer, [('magnitude', 'bold_file')])])

    # Remove non-steady-state volumes
    # XXX: This probably introduces a mismatch in the number of volumes of the data and the
    # motion xfm.
    remove_mag_nss = pe.Node(
        RemoveNSS(skip_vols=skip_vols),
        name='remove_mag_nss',
    )
    workflow.connect([(stc_buffer, remove_mag_nss, [('bold_file', 'in_file')])])

    remove_phase_nss = pe.Node(
        RemoveNSS(skip_vols=skip_vols),
        name='remove_phase_nss',
    )
    workflow.connect([(denoise_buffer, remove_phase_nss, [('phase', 'in_file')])])

    # Unwrap phase with warpkit (multi-echo) or ROMEO (single-echo)
    if multiecho:
        # Rescale phase to 0 to 2*pi before unwrapping
        phase_to_radians2 = pe.Node(
            Scale2Radians(scale='2pi'),
            name='phase_to_radians2',
        )
        workflow.connect([(remove_phase_nss, phase_to_radians2, [('out_file', 'in_file')])])

        # Convert metadata dictionaries to JSON files
        metadata_to_jsons = pe.Node(
            DictToJSON(),
            name='metadata_to_jsons',
        )
        workflow.connect([(inputnode, metadata_to_jsons, [('metadata', 'in_dicts')])])

        unwrap_phase = pe.Node(
            WarpkitUnwrap(
                noise_frames=0,
                debug=False,
                wrap_limit=False,
                n_cpus=config.nipype.omp_nthreads,
            ),
            name='unwrap_phase',
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([
            (phase_to_radians2, unwrap_phase, [('out_file', 'phase')]),
            (metadata_to_jsons, unwrap_phase, [('json_files', 'metadata')]),
        ])  # fmt:skip
    else:
        # ROMEO uses data in radians (-pi to pi)
        unwrap_phase = pe.Node(
            ROMEOUnwrap(
                no_scale=True,
                echo_times=bold_metadata['EchoTime'] * 1000,
                mask='nomask',
            ),
            name='unwrap_phase',
        )
        workflow.connect([(remove_phase_nss, unwrap_phase, [('out_file', 'phase')])])

    workflow.connect([(remove_mag_nss, unwrap_phase, [('out_file', 'magnitude')])])

    # Warp magnitude and phase data to BOLD reference space
    mag_boldref_wf = init_bold_volumetric_resample_wf(
        metadata=bold_metadata,
        fieldmap_id=fmapid,
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        jacobian='fmap-jacobian' not in config.workflow.ignore,
        name='mag_boldref_wf',
    )
    workflow.connect([
        (inputnode, mag_boldref_wf, [
            ('hmc', 'motion_xfm'),
            ('boldref2fmap', 'boldref2fmap_xfm'),
            ('bold_mask_native', 'bold_ref_file'),
        ]),
    ])  # fmt:skip

    workflow.connect([
        (remove_mag_nss, mag_boldref_wf, [('out_file', 'inputnode.bold_file')]),
        (inputnode, mag_boldref_wf, [
            ('fmap_ref', 'inputnode.fmap_ref'),
            ('fmap_coeff', 'inputnode.fmap_coeff'),
            ('fmap_id', 'inputnode.fmap_id'),
        ]),
    ])  # fmt:skip

    # Warp to boldref space (motion correction)
    phase_boldref_wf = init_bold_volumetric_resample_wf(
        metadata=bold_metadata,
        fieldmap_id=None,  # Ignore the field map for phase data
        omp_nthreads=omp_nthreads,
        mem_gb=mem_gb,
        jacobian=False,
        name='phase_boldref_wf',
    )
    workflow.connect([
        (inputnode, phase_boldref_wf, [
            ('hmc', 'motion_xfm'),
            ('boldref2fmap', 'boldref2fmap_xfm'),
            ('bold_mask_native', 'bold_ref_file'),
        ]),
        (unwrap_phase, phase_boldref_wf, [('unwrapped', 'inputnode.bold_file')]),
    ])  # fmt:skip

    if config.workflow.retroicor:
        # Run RETROICOR on the magnitude and phase data
        # After rescaling + unwrapping
        # TODO: Load physio data
        raise NotImplementedError('RETROICOR is not yet implemented.')

    if config.workflow.regression_method:
        # Now denoise the BOLD data using phase regression
        phase_regression_wf = init_phase_regression_wf(
            name_source=bold_file,
            metadata=bold_metadata,
        )
        workflow.connect([
            (inputnode, phase_regression_wf, [('bold_mask_native', 'bold_mask')]),
            (phase_boldref_wf, phase_regression_wf, [('outputnode.bold_file', 'inputnode.phase')]),
            (mag_boldref_wf, phase_regression_wf, [
                ('outputnode.bold_file', 'inputnode.magnitude'),
            ]),
        ])  # fmt:skip

    # Calculate phase jolt, jump, and/or laplacian files
    # TODO: Apply transforms to jolt, jump, and laplacian files to target spaces
    # TODO: Create figures for derivatives, including carpet plots and registration plots
    if config.workflow.jolt:
        # Calculate phase jolt
        calc_jolt = pe.Node(
            LayNiiPhaseJolt(phase_jump=False),
            name='calc_jolt',
        )
        workflow.connect([(remove_phase_nss, calc_jolt, [('out_file', 'in_file')])])

    if config.workflow.jump:
        # Calculate phase jump
        calc_jump = pe.Node(
            LayNiiPhaseJolt(phase_jump=True),
            name='calc_jump',
        )
        workflow.connect([(remove_phase_nss, calc_jump, [('out_file', 'in_file')])])

    if config.workflow.laplacian:
        # Run Laplacian
        calc_laplacian = pe.Node(
            LayNiiPhaseLaplacian(),
            name='calc_laplacian',
        )
        workflow.connect([(remove_phase_nss, calc_laplacian, [('out_file', 'in_file')])])

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
    workflow.connect([
        (inputnode, bold_confounds_wf, [('bold_mask_native', 'bold_mask')]),
        (phase_boldref_wf, bold_confounds_wf, [('outputnode.bold_file', 'inputnode.phase')]),
    ])  # fmt:skip

    ds_confounds = pe.Node(
        DerivativesDataSink(
            desc='confounds',
            suffix='timeseries',
            extension='.tsv',
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

    # Warp derivatives to boldref space
    boldref_buffer = pe.Node(
        niu.IdentityInterface(
            fields=[
                'jolt',
                'jump',
                'laplacian',
                'phaseDenoised',
                'phaseNoise',
                'unwrapped',
            ],
        ),
        name='boldref_buffer',
    )
    boldref_derivatives = []
    if config.workflow.jolt:
        boldref_derivatives.append('jolt')
        jolt_boldref_wf = init_bold_volumetric_resample_wf(
            metadata=bold_metadata,
            fieldmap_id=None,  # Ignore the field map for phase data
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            jacobian=False,
            name='jolt_boldref_wf',
        )
        workflow.connect([
            (inputnode, jolt_boldref_wf, [
                ('hmc', 'motion_xfm'),
                ('boldref2fmap', 'boldref2fmap_xfm'),
                ('bold_mask_native', 'bold_ref_file'),
            ]),
            (calc_jolt, jolt_boldref_wf, [('out_file', 'inputnode.bold_file')]),
            (jolt_boldref_wf, boldref_buffer, [('outputnode.bold_file', 'jolt')]),
        ])  # fmt:skip

    if config.workflow.jump:
        boldref_derivatives.append('jump')
        jump_boldref_wf = init_bold_volumetric_resample_wf(
            metadata=bold_metadata,
            fieldmap_id=None,  # Ignore the field map for phase data
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            jacobian=False,
            name='jump_boldref_wf',
        )
        workflow.connect([
            (inputnode, jump_boldref_wf, [
                ('hmc', 'motion_xfm'),
                ('boldref2fmap', 'boldref2fmap_xfm'),
                ('bold_mask_native', 'bold_ref_file'),
            ]),
            (calc_jump, jump_boldref_wf, [('out_file', 'inputnode.bold_file')]),
            (jump_boldref_wf, boldref_buffer, [('outputnode.bold_file', 'jump')]),
        ])  # fmt:skip

    if config.workflow.laplacian:
        boldref_derivatives.append('laplacian')
        laplacian_boldref_wf = init_bold_volumetric_resample_wf(
            metadata=bold_metadata,
            fieldmap_id=None,  # Ignore the field map for phase data
            omp_nthreads=omp_nthreads,
            mem_gb=mem_gb,
            jacobian=False,
            name='laplacian_boldref_wf',
        )
        workflow.connect([
            (inputnode, laplacian_boldref_wf, [
                ('hmc', 'motion_xfm'),
                ('boldref2fmap', 'boldref2fmap_xfm'),
                ('bold_mask_native', 'bold_ref_file'),
            ]),
            (calc_laplacian, laplacian_boldref_wf, [('out_file', 'inputnode.bold_file')]),
            (laplacian_boldref_wf, boldref_buffer, [('outputnode.bold_file', 'laplacian')]),
        ])  # fmt:skip

    if config.workflow.regression_method:
        boldref_derivatives.append('phaseDenoised')
        boldref_derivatives.append('phaseNoise')
        workflow.connect([
            (phase_regression_wf, boldref_buffer, [
                ('outputnode.denoised_magnitude', 'phaseDenoised'),
                ('outputnode.phase', 'phaseNoise'),
            ]),
        ])  # fmt:skip

    if config.workflow.unwrapped_phase:
        boldref_derivatives.append('unwrapped')
        workflow.connect([
            (phase_boldref_wf, boldref_buffer, [('outputnode.bold_file', 'unwrapped')]),
        ])  # fmt:skip

    for boldref_derivative in boldref_derivatives:
        # Carpet plot
        deriv_carpetplot_wf = init_carpetplot_wf(
            mem_gb=mem_gb,
            metadata=bold_metadata,
            cifti_output=config.workflow.cifti_output,
            name=f'{boldref_derivative}_carpet_wf',
        )
        deriv_carpetplot_wf.inputs.inputnode.desc = boldref_derivative
        workflow.connect([
            (inputnode, deriv_carpetplot_wf, [
                ('bold_mask_native', 'bold_mask'),
                ('boldref2anat_xfm', 'boldref2anat_xfm'),
                ('mni2009c2anat_xfm', 'inputnode.std2anat_xfm'),
            ]),
            (boldref_buffer, deriv_carpetplot_wf, [(boldref_derivative, 'inputnode.bold')]),
            (bold_confounds_wf, deriv_carpetplot_wf, [
                ('outputnode.confounds_file', 'inputnode.confounds_file'),
            ]),
        ])  # fmt:skip

    # TODO: Wrangle desired output spaces and warp derivatives to them
    boldref_out = bool(nonstd_spaces.intersection(('func', 'run', 'bold', 'boldref', 'sbref')))

    if boldref_out:
        for boldref_derivative in boldref_derivatives:
            # jolt, jump, laplacian, unwrapped phase, denoising summary maps
            ds_deriv_boldref = pe.Node(
                DerivativesDataSink(
                    desc=boldref_derivative,
                    part='phase',
                    suffix='bold',
                ),
                name=f'ds_{boldref_derivative}_boldref',
            )
            workflow.connect([
                (boldref_buffer, ds_deriv_boldref, [(boldref_derivative, 'in_file')]),
            ])  # fmt:skip

    # Full derivatives, including resampled BOLD series
    if nonstd_spaces.intersection(('anat', 'T1w')):
        ...

    if spaces.cached.get_spaces(nonstandard=False, dim=(3,)):
        ...

    if config.workflow.run_reconall and freesurfer_spaces:
        ...

    if config.workflow.cifti_output:
        ...

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
