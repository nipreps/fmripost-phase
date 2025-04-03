"""Denoising workflows."""


def init_nordic_wf(
    mem_gb: float,
    name: str = 'nordic_wf',
):
    """Build a workflow to apply thermal denoising with NORDIC."""
    from fmriprep.utils.bids import dismiss_echo
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.reportlets.masks import ROIsPlot

    from fmripost_phase.interfaces.bids import DerivativesDataSink
    from fmripost_phase.interfaces.denoising import NORDIC
    from fmripost_phase.utils.utils import clean_datasinks

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude',
                'phase',
                'magnitude_norf',
                'phase_norf',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude',
                'phase',
            ],
        ),
        name='outputnode',
    )

    denoise = pe.Node(
        NORDIC(
            algorithm='nordic',
            save_gfactor_map=True,
        ),
        name='denoise',
        mem_gb=mem_gb,
    )
    workflow.connect([
        (inputnode, denoise, [
            ('magnitude', 'magnitude'),
            ('phase', 'phase'),
            ('magnitude_norf', 'magnitude_norf'),
            ('phase_norf', 'phase_norf'),
        ]),
        (denoise, outputnode, [
            ('magnitude', 'magnitude'),
            ('phase', 'phase'),
        ]),
    ])  # fmt:skip

    # Generate reportlet (G-Factor)
    gfactor_plot = pe.Node(
        ROIsPlot(),
        name='gfactor_plot',
        mem_gb=mem_gb,
    )
    ds_report_gfactor = pe.Node(
        DerivativesDataSink(desc='gfactor', dismiss_entities=dismiss_echo()),
        name='ds_report_gfactor',
        run_without_submitting=True,
        mem_gb=mem_gb,
    )
    workflow.connect([
        (denoise, gfactor_plot, [('gfactor', 'in_file')]),
        (gfactor_plot, ds_report_gfactor, [('out_report', 'in_file')]),
    ])  # fmt:skip

    # Generate reportlet (Number of components removed)
    n_components_removed_plot = pe.Node(
        ROIsPlot(),
        name='n_components_removed_plot',
        mem_gb=mem_gb,
    )
    ds_report_n_components_removed = pe.Node(
        DerivativesDataSink(desc='componentsremoved', dismiss_entities=dismiss_echo()),
        name='ds_report_n_components_removed',
        run_without_submitting=True,
        mem_gb=mem_gb,
    )
    workflow.connect([
        (denoise, n_components_removed_plot, [('n_components_removed', 'in_file')]),
        (n_components_removed_plot, ds_report_n_components_removed, [('out_report', 'in_file')]),
    ])  # fmt:skip

    # Generate reportlet (Noise)
    noise_plot = pe.Node(
        ROIsPlot(),
        name='noise_plot',
        mem_gb=mem_gb,
    )
    ds_report_noise = pe.Node(
        DerivativesDataSink(desc='thermalnoise', dismiss_entities=dismiss_echo()),
        name='ds_report_noise',
        run_without_submitting=True,
        mem_gb=mem_gb,
    )
    workflow.connect([
        (denoise, noise_plot, [('noise', 'in_file')]),
        (noise_plot, ds_report_noise, [('out_report', 'in_file')]),
    ])  # fmt:skip

    return clean_datasinks(workflow)


def init_dwidenoise_wf(
    mem_gb: float,
    name: str = 'dwidenoise_wf',
):
    """Build a workflow to apply thermal denoising with dwidenoise."""
    from fmriprep.utils.bids import dismiss_echo
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.reportlets.masks import ROIsPlot

    from fmripost_phase import config
    from fmripost_phase.interfaces.bids import DerivativesDataSink
    from fmripost_phase.interfaces.complex import (
        ComplexToMagnitude,
        ComplexToPhase,
        PolarToComplex,
    )
    from fmripost_phase.interfaces.denoising import DWIDenoise
    from fmripost_phase.utils.utils import clean_datasinks

    omp_nthreads = config.nipype.omp_nthreads

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude',
                'phase',
                'magnitude_norf',
                'phase_norf',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'magnitude',
                'phase',
            ],
        ),
        name='outputnode',
    )

    # Phase data are already in radians (-pi to pi)
    combine_complex = pe.Node(
        PolarToComplex(),
        name='combine_complex',
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (inputnode, combine_complex, [
            ('magnitude', 'mag_file'),
            ('phase', 'phase_file'),
        ]),
    ])  # fmt:skip

    denoise = pe.Node(
        DWIDenoise(),
        name='denoise',
        mem_gb=mem_gb,
    )
    workflow.connect([(combine_complex, denoise, [('out_file', 'in_file')])])

    complex_to_magnitude = pe.Node(
        ComplexToMagnitude(),
        name='complex_to_magnitude',
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (denoise, complex_to_magnitude, [('out_file', 'complex_file')]),
        (complex_to_magnitude, outputnode, [('out_file', 'magnitude')]),
    ])  # fmt:skip

    complex_to_phase = pe.Node(
        ComplexToPhase(),
        name='complex_to_phase',
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (denoise, complex_to_phase, [('out_file', 'complex_file')]),
        (complex_to_phase, outputnode, [('out_file', 'phase')]),
    ])  # fmt:skip

    # I think the noise map needs to be rescaled (divide by sqrt(2))
    rescale_noise = pe.Node(
        niu.Apply(
            expr='i / sqrt(2)',
            output_name='out_file',
        ),
        name='rescale_noise',
    )
    workflow.connect([(denoise, rescale_noise, [('noise', 'in_file')])])

    # Generate reportlet (Noise)
    noise_plot = pe.Node(
        ROIsPlot(),
        name='noise_plot',
        mem_gb=mem_gb,
    )
    ds_report_noise = pe.Node(
        DerivativesDataSink(desc='thermalnoise', dismiss_entities=dismiss_echo()),
        name='ds_report_noise',
        run_without_submitting=True,
        mem_gb=mem_gb,
    )
    workflow.connect([
        (rescale_noise, noise_plot, [('out_file', 'in_file')]),
        (noise_plot, ds_report_noise, [('out_report', 'in_file')]),
    ])  # fmt:skip

    return clean_datasinks(workflow)
