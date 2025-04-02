"""Denoising workflows."""


def init_denoise_wf(
    mem_gb: float,
    name: str = 'denoise_wf',
):
    """Build a workflow to apply thermal denoising."""
    from fmriprep.utils.bids import dismiss_echo
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.reportlets.masks import ROIsPlot

    from fmripost_phase import config
    from fmripost_phase.interfaces.bids import DerivativesDataSink
    from fmripost_phase.interfaces.denoising import NORDIC, DWIDenoise
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

    if config.workflow.thermal_denoise_method == 'nordic':
        denoise = pe.Node(
            NORDIC(
                algorithm='nordic',
                save_gfactor_map=True,
            ),
            name='denoise',
            mem_gb=mem_gb,
        )
    else:
        # TODO: Combine magnitude and phase into complex data
        denoise = pe.Node(
            DWIDenoise(),
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
