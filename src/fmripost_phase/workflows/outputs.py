"""Writing out derivative files."""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from fmripost_phase import config
from fmripost_phase.config import DEFAULT_MEMORY_MIN_GB
from fmripost_phase.interfaces import DerivativesDataSink
from fmripost_phase.interfaces.bids import BIDSURI
from fmripost_phase.utils.bids import dismiss_echo


def init_ds_bold_native_wf(
    *,
    bids_root: str,
    output_dir: str,
    multiecho: bool,
    bold_output: bool,
    all_metadata: list[dict],
    name='ds_bold_native_wf',
) -> pe.Workflow:
    metadata = all_metadata[0]
    timing_parameters = prepare_timing_parameters(metadata)

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_files',
                'bold',
                'bold_echos',
                't2star',
                # Transforms previously used to generate the outputs
                'motion_xfm',
                'boldref2fmap_xfm',
            ]
        ),
        name='inputnode',
    )

    sources = pe.Node(
        BIDSURI(
            numinputs=3,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )
    workflow.connect([
        (inputnode, sources, [
            ('source_files', 'in1'),
            ('motion_xfm', 'in2'),
            ('boldref2fmap_xfm', 'in3'),
        ]),
    ])  # fmt:skip

    if bold_output:
        ds_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='preproc',
                compress=True,
                SkullStripped=multiecho,
                TaskName=metadata.get('TaskName'),
                dismiss_entities=dismiss_echo(),
                **timing_parameters,
            ),
            name='ds_bold',
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_bold, [
                ('source_files', 'source_file'),
                ('bold', 'in_file'),
            ]),
            (sources, ds_bold, [('out', 'Sources')]),
        ])  # fmt:skip

    if bold_output and multiecho:
        t2star_meta = {
            'Units': 's',
            'EstimationReference': 'doi:10.1002/mrm.20900',
            'EstimationAlgorithm': 'monoexponential decay model',
        }
        ds_t2star = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space='boldref',
                suffix='T2starmap',
                compress=True,
                dismiss_entities=dismiss_echo(),
                **t2star_meta,
            ),
            name='ds_t2star_bold',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_t2star, [
                ('source_files', 'source_file'),
                ('t2star', 'in_file'),
            ]),
            (sources, ds_t2star, [('out', 'Sources')]),
        ])  # fmt:skip

    if echo_output:
        ds_bold_echos = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='preproc',
                compress=True,
                SkullStripped=False,
                TaskName=metadata.get('TaskName'),
                **timing_parameters,
            ),
            iterfield=['source_file', 'in_file', 'meta_dict'],
            name='ds_bold_echos',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_bold_echos.inputs.meta_dict = [{'EchoTime': md['EchoTime']} for md in all_metadata]
        workflow.connect([
            (inputnode, ds_bold_echos, [
                ('source_files', 'source_file'),
                ('bold_echos', 'in_file'),
            ]),
        ])  # fmt:skip

    return workflow
