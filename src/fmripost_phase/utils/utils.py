"""Utility functions for fmriprep_phase."""


def _get_wf_name(bold_fname, prefix):
    """Derive the workflow name for supplied BOLD file.

    >>> _get_wf_name("/completely/made/up/path/sub-01_task-nback_bold.nii.gz", "aroma")
    'aroma_task_nback_wf'
    >>> _get_wf_name(
    ...     "/completely/made/up/path/sub-01_task-nback_run-01_echo-1_bold.nii.gz",
    ...     "preproc",
    ... )
    'preproc_task_nback_run_01_echo_1_wf'

    """
    from nipype.utils.filemanip import split_filename

    fname = split_filename(bold_fname)[1]
    fname_nosub = '_'.join(fname.split('_')[1:-1])
    return f"{prefix}_{fname_nosub.replace('-', '_')}_wf"
