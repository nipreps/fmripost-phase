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
        (highcor, compcor_plot, [('metadata_file', 'in1')]),
        (compcor_plot, ds_report_compcor, [('out_file', 'in_file')]),
    ])  # fmt:skip

    return clean_datasinks(workflow)
