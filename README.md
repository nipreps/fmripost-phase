# fMRIPost-Phase

Preprocessing of fMRI phase data using BIDS derivatives from magnitude preprocessing.

## Expected outputs

```
sub-<label>/[ses-<label>/]
    func/
        # Unwrapped phase data
        <source_entities>[_space-<label>]_desc-unwrapped_bold.nii.gz
        # First derivative of phase image
        <source_entities>[_space-<label>]_desc-jump_bold.nii.gz
        # Second derivative of phase image
        <source_entities>[_space-<label>]_desc-jolt_bold.nii.gz
        # Confounds extracted from the phase data
        <source_entities>_desc-confounds_timeseries.tsv
        <source_entities>_desc-confounds_timeseries.json
```
