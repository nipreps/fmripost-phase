"""Interfaces for denoising."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.mrtrix3.base import MRTrix3Base, MRTrix3BaseInputSpec


class _NORDICInputSpec(BaseInterfaceInputSpec):
    magnitude = File(exists=True, mandatory=True, desc='Input magnitude file')
    phase = File(exists=True, mandatory=False, desc='Input phase file')
    magnitude_norf = File(exists=True, mandatory=False, desc='Input magnitude noise file')
    phase_norf = File(exists=True, mandatory=False, desc='Input phase noise file')
    algorithm = traits.Enum(
        'nordic',
        'mppca',
        'gfactor+mppca',
        desc='Algorithm to use for denoising',
    )
    temporal_phase = traits.Int(
        1,
        usedefault=True,
        desc='Temporal phase to use for denoising',
    )
    phase_filter_width = traits.Int(
        3,
        usedefault=True,
        desc='Width of phase filter',
    )
    factor_error = traits.Float(
        1,
        usedefault=True,
        desc='Factor for error estimation',
    )
    full_dynamic_range = traits.Bool(
        True,
        desc='Whether to use full dynamic range',
    )
    patch_overlap_gfactor = traits.Int(
        2,
        usedefault=True,
        desc='Patch overlap for g-factor calculation',
    )
    kernel_size_gfactor = traits.Int(
        None,
        desc='Kernel size for g-factor calculation',
    )
    patch_overlap_pca = traits.Int(
        2,
        usedefault=True,
        desc='Patch overlap for PCA',
    )
    kernel_size_pca = traits.Int(
        None,
        desc='Kernel size for PCA',
    )
    phase_slice_average_for_kspace_centering = traits.Bool(
        False,
        usedefault=True,
        desc='Whether to use phase slice average for k-space centering',
    )
    save_gfactor_map = traits.Bool(
        True,
        usedefault=True,
        desc='Whether to save g-factor map',
    )
    soft_thrs = traits.Float(
        'auto',
        usedefault=True,
        desc='Soft threshold value',
    )
    debug = traits.Bool(
        False,
        usedefault=True,
        desc='Whether to run in debug mode',
    )
    scale_patches = traits.Bool(
        False,
        usedefault=True,
        desc='Whether to scale patches',
    )
    patch_average = traits.Bool(
        False,
        usedefault=True,
        desc='Whether to average patches',
    )
    llr_scale = traits.Float(
        1,
        usedefault=True,
        desc='Scale factor for low-rank regularization',
    )


class _NORDICOutputSpec(TraitedSpec):
    magnitude = File(exists=True, desc='Denoised magnitude file')
    phase = File(exists=True, desc='Denoised phase file')
    gfactor = File(exists=True, desc='G-factor file')
    n_components_removed = File(exists=True, desc='Number of components removed file')
    noise = File(exists=True, desc='Noise file')
    measured_noise = traits.Float(desc='Measured noise level')


class NORDIC(SimpleInterface):
    """Run NORDIC."""

    input_spec = _NORDICInputSpec
    output_spec = _NORDICOutputSpec

    def _run_interface(self, runtime):
        import os

        from fmripost_phase.utils.nordic import run_nordic

        psafkc = self.inputs.phase_slice_average_for_kspace_centering

        measured_noise = run_nordic(
            mag_file=self.inputs.magnitude,
            pha_file=self.inputs.phase,
            mag_norf_file=self.inputs.magnitude_norf,
            pha_norf_file=self.inputs.phase_norf,
            out_dir=runtime.cwd,
            factor_error=self.inputs.factor_error,
            full_dynamic_range=self.inputs.full_dynamic_range,
            temporal_phase=self.inputs.temporal_phase,
            algorithm=self.inputs.algorithm,
            patch_overlap_gfactor=self.inputs.patch_overlap_gfactor,
            kernel_size_gfactor=self.inputs.kernel_size_gfactor,
            patch_overlap_pca=self.inputs.patch_overlap_pca,
            kernel_size_pca=self.inputs.kernel_size_pca,
            phase_slice_average_for_kspace_centering=psafkc,
            phase_filter_width=self.inputs.phase_filter_width,
            save_gfactor_map=self.inputs.save_gfactor_map,
            soft_thrs=self.inputs.soft_thrs,
            debug=self.inputs.debug,
            scale_patches=self.inputs.scale_patches,
            patch_average=self.inputs.patch_average,
            llr_scale=self.inputs.llr_scale,
        )

        self._results['magnitude'] = os.path.join(self.inputs.out_dir, 'magn.nii.gz')

        if self.inputs.phase:
            self._results['phase'] = os.path.join(self.inputs.out_dir, 'phase.nii.gz')

        if self.inputs.save_gfactor_map:
            self._results['gfactor'] = os.path.join(self.inputs.out_dir, 'gfactor.nii.gz')

        self._results['n_components_removed'] = os.path.join(
            self.inputs.out_dir,
            'n_components_removed.nii.gz',
        )

        self._results['noise'] = os.path.join(self.inputs.out_dir, 'noise.nii.gz')
        self._results['measured_noise'] = measured_noise

        return runtime


class DWIDenoiseInputSpec(MRTrix3BaseInputSpec):
    in_file = File(exists=True, argstr='%s', position=-2, mandatory=True, desc='input DWI image')
    mask = File(exists=True, argstr='-mask %s', position=1, desc='mask image')
    extent = traits.Tuple(
        (traits.Int, traits.Int, traits.Int),
        argstr='-extent %d,%d,%d',
        desc='set the window size of the denoising filter. (default = 5,5,5)',
    )
    noise_image = File(
        argstr='-noise %s',
        name_template='%s_noise.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        desc='the output noise map',
    )
    out_file = File(
        name_template='%s_denoised.nii.gz',
        name_source=['in_file'],
        keep_extension=False,
        argstr='%s',
        position=-1,
        desc='the output denoised DWI image',
    )


class DWIDenoiseOutputSpec(TraitedSpec):
    noise_image = File(desc='the output noise map', exists=True)
    out_file = File(desc='the output denoised DWI image', exists=True)


class DWIDenoise(MRTrix3Base):
    """
    Denoise DWI data and estimate the noise level based on the optimal
    threshold for PCA.

    DWI data denoising and noise map estimation by exploiting data redundancy
    in the PCA domain using the prior knowledge that the eigenspectrum of
    random covariance matrices is described by the universal Marchenko Pastur
    distribution.

    Important note: image denoising must be performed as the first step of the
    image processing pipeline. The routine will fail if interpolation or
    smoothing has been applied to the data prior to denoising.

    Note that this function does not correct for non-Gaussian noise biases.

    For more information, see
    <https://mrtrix.readthedocs.io/en/latest/reference/commands/dwidenoise.html>

    """

    _cmd = 'dwidenoise'
    input_spec = DWIDenoiseInputSpec
    output_spec = DWIDenoiseOutputSpec
