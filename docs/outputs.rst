.. include:: links.rst

.. _outputs:

---------------------------
Outputs of *fMRIPost-Phase*
---------------------------

*fMRIPost-Phase* outputs conform to the :abbr:`BIDS (brain imaging data structure)`
Derivatives specification (see `BIDS Derivatives`_, along with the
upcoming `BEP 011`_ and `BEP 012`_).
*fMRIPost-Phase* generates three broad classes of outcomes:

1.  **Visual QA (quality assessment) reports**:
    One :abbr:`HTML (hypertext markup language)` per subject,
    that allows the user a thorough visual assessment of the quality
    of processing and ensures the transparency of *fMRIPost-Phase* operation.

2.  **ICA outputs**:
    Outputs from the independent component analysis (ICA).
    For example, the mixing matrix and component weight maps.

3.  **Derivatives (denoised data)**:
    Denoised fMRI data in the requested output spaces and resolutions.

4.  **Confounds**:
    Time series of ICA components classified as noise.


Layout
------

Assuming fMRIPost-Phase is invoked with::

    fmripost_phase <input_dir>/ <output_dir>/ participant [OPTIONS]

The outputs will be a `BIDS Derivatives`_ dataset of the form::

    <output_dir>/
      logs/
      sub-<label>/
      sub-<label>.html
      dataset_description.json
      .bidsignore

For each participant in the dataset,
a directory of derivatives (``sub-<label>/``)
and a visual report (``sub-<label>.html``) are generated.
The log directory contains `citation boilerplate`_ text.
``dataset_description.json`` is a metadata file in which fMRIPost-Phase
records metadata recommended by the BIDS standard.


Visual Reports
--------------

*fMRIPost-Phase* outputs summary reports,
written to ``<output dir>/fmripost_phase/sub-<label>.html``.
These reports provide a quick way to make visual inspection of the results easy.


Derivatives of *fMRIPost-Phase* (denoised data)
-----------------------------------------------

Derivative data are written to
``<output dir>/sub-<label>/``.
The `BIDS Derivatives`_ specification describes the naming and metadata conventions we follow.

Functional derivatives
~~~~~~~~~~~~~~~~~~~~~~

**Unwrapped phase data**.
Unwrapped phase data are stored in the ``func/`` subfolder,
if the ``--unwrap`` flag is used::

  sub-<label>/
    func/
      <source_entities>_part-phase_desc-unwrapped_bold.nii.gz
      <source_entities>_part-phase_desc-unwrapped_bold.json

**Phase regression outputs**.
Phase regression confounds are stored in the ``func/`` subfolder,
if ``--regression-method`` is not set to ``none``::

  sub-<label>/
    func/
      # The phase denoised BOLD series
      <source_entities>_part-mag_desc-phaseDenoised_bold.nii.gz
      <source_entities>_part-mag_desc-phaseDenoised_bold.json
      # Fitted phase time series from the phase regression
      # This can be combined with other confounds for denoising
      <source_entities>_desc-fitted_timeseries.nii.gz
      <source_entities>_desc-fitted_timeseries.json

**LayNII phase outputs**.
LayNII phase outputs are stored in the ``func/`` subfolder::

  sub-<label>/
    func/
      # If --jump is enabled
      <source_entities>_part-phase_desc-jump_bold.nii.gz
      <source_entities>_part-phase_desc-jump_bold.json

      # If --jolt is enabled
      <source_entities>_part-phase_desc-jolt_bold.nii.gz
      <source_entities>_part-phase_desc-jolt_bold.json

**Complex-ICA derivatives**.
Complex-ICA outputs from GIFT are stored in the ``func/`` subfolder,
if ``--gift-dimensionality`` is not set to ``0``::

  sub-<label>/
    func/
      <source_entities>_part-mag_desc-gift_mixing.tsv
      <source_entities>_part-mag_desc-gift_mixing.json
      <source_entities>_part-phase_desc-gift_mixing.tsv
      <source_entities>_part-phase_desc-gift_mixing.json
      <source_entities>_part-mag_desc-gift_components.nii.gz
      <source_entities>_part-mag_desc-gift_components.json
      <source_entities>_part-phase_desc-gift_components.nii.gz
      <source_entities>_part-phase_desc-gift_components.json

**Extracted confounding time series**.
For each :abbr:`BOLD (blood-oxygen level dependent)` run processed with *fMRIPost-Phase*,
an accompanying *confounds* file will be generated.

*fMRIPost-Phase* outputs a set of confounds that can be used to denoise the data.
These are stored in a TSV file (``desc-confounds_timeseries.tsv``) and a JSON file
(``desc-confounds_timeseries.json``) that contains metadata about the confounds.

Confounds_ are saved as a :abbr:`TSV (tab-separated value)` file::

  sub-<label>/
    func/
      <source_entities>_desc-confounds_timeseries.tsv
      <source_entities>_desc-confounds_timeseries.json

**CompCor confounds**.
:abbr:`CompCor (Component Based Noise Correction)` is a :abbr:`PCA (principal component analysis)`,
hence component-based, noise pattern recognition method.
In the method, principal components are calculated within an :abbr:`ROI (Region of Interest)`
that is unlikely to include signal related to neuronal activity, such as :abbr:`CSF (cerebro-spinal fluid)`
and :abbr:`WM (white matter)` masks.
Signals extracted from CompCor components can be further regressed out from the fMRI data with a
denoising procedure [Behzadi2007]_.

- ``h_comp_cor_XX``: noise components as calculated using HighCor.
- ``highcor``: mean time series from the high-variance voxels identified by HighCor.


Confounds and "carpet"-plot on the visual reports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The visual reports provide several sections per task and run to aid designing
a denoising strategy for subsequent analysis.
Some of the estimated confounds are plotted with a "carpet" visualization of the
:abbr:`BOLD (blood-oxygen level-dependent)` time series [Power2016]_.
An example of these plots follows:

See implementation on :mod:`~fmripost_phase.workflows.bold.confounds.init_bold_confs_wf`.
