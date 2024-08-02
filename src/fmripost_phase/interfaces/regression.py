"""Interfaces for phase regression."""

import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)


class _ODRFitInputSpec(BaseInterfaceInputSpec):
    phase = File(exists=True, mandatory=True, desc="phase image")
    magnitude = File(exists=True, mandatory=True, desc="magnitude image")
    mask = File(exists=True, mandatory=True, desc="mask")
    TR = traits.Float(mandatory=True, desc="Repetition time of scan")
    noise_filter = traits.Float(mandatory=True, desc="high-pass filter threshold")


class _ODRFitOutputSpec(TraitedSpec):
    sim = File(exists=True, mandatory=True, desc="")
    filt = File(exists=True, mandatory=True, desc="")
    residuals = File(exists=True, mandatory=True, desc="")
    xres = File(exists=True, mandatory=True, desc="")
    yres = File(exists=True, mandatory=True, desc="")
    xplus = File(exists=True, mandatory=True, desc="")
    stdp = File(exists=True, mandatory=True, desc="")
    stdm = File(exists=True, mandatory=True, desc="")
    r2 = File(exists=True, mandatory=True, desc="")
    estimate = File(exists=True, mandatory=True, desc="")


class ODRFit(SimpleInterface):
    """Regress phase from magnitude with orthogonal distance regression."""

    input_spec = _ODRFitInputSpec
    output_spec = _ODRFitOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        import scipy.odr.odrpack as odr
        from nipype.utils.filemanip import fname_presuffix

        # Load phase, magnitude and mask
        mag_img = nb.load(self.inputs.magnitude)
        mag_data = mag_img.get_fdata()

        phase_img = nb.load(self.inputs.phase)
        phase_data = phase_img.get_fdata()

        mask_img = nb.load(self.inputs.mask)
        mask_data = mask_img.get_fdata()

        # Set TR and filter threshold
        TR = float(self.inputs.TR)
        noise_lb = float(self.inputs.noise_filter)

        # Create variables were the outputs will be saved
        out_shape = np.array(mag_data.shape)  # 4D variable with the shape
        nt = mag_data.shape[-1]

        # Array of zeros with length = number of voxels
        scales = np.zeros(np.prod(out_shape[0:-1]))
        # 2D matrix of zeros with shape voxels x timepoints
        filt = np.zeros((np.prod(out_shape[0:-1]), nt))
        sim = np.zeros_like(filt)
        residuals = np.zeros_like(filt)

        delta = np.zeros_like(filt)
        eps = np.zeros_like(filt)
        xshift = np.zeros_like(filt)
        stdm = np.zeros(out_shape[0:-1])
        stdp = np.zeros(out_shape[0:-1])
        r2 = np.zeros_like(scales)

        estimate = np.zeros_like(filt)

        mag_data = np.array(mag_data)

        # Create model
        linearfit = odr.Model(_linear)

        # Creates a noise mask that takes only those values greater than noise_lb
        freqs = np.linspace(-1.0, 1.0, nt) / (2 * TR)
        noise_mask = np.fft.fftshift(1.0 * (abs(freqs) > noise_lb))

        # Estimates standard deviations of magnitude and phase
        for i_x in range(mag_data.shape[0]):
            temp = mag_data[i_x, :, :, :]
            stdm[i_x, :, :] = np.std(np.fft.ifft(np.fft.fft(temp) * noise_mask), -1)
            temp = phase_data[i_x, :, :, :]
            stdp[i_x, :, :] = np.std(np.fft.ifft(np.fft.fft(temp) * noise_mask), -1)

        # Reshape variables into a single column
        # Reshapes variable intro 2D matrix of voxels x timepoints
        mag_data = np.reshape(mag_data, (-1, nt))
        phase_data = np.reshape(phase_data, (-1, nt))
        # Reshapes variable intro array of length = number of voxels
        stdm = np.reshape(stdm, (-1,))
        stdp = np.reshape(stdp, (-1,))
        mask_data = np.reshape(mask_data, (-1,))

        for i_x in range(mag_data.shape[0]):
            if mask_data[i_x]:
                design = phase_data[i_x, :]
                ests = [stdm[i_x] / stdp[i_x], 1.0]
                mydata = odr.RealData(design, mag_data[i_x, :], sx=stdp[i_x], sy=stdm[i_x])
                odr_obj = odr.ODR(mydata, linearfit, beta0=ests, maxit=600)
                res = odr_obj.run()
                est = res.y

                r2[i_x] = 1.0 - (
                    np.sum((mag_data[i_x, :] - est) ** 2) / np.sum((mag_data[i_x, :]) ** 2)
                )

                # take out scaled phase signal and re-mean may need correction
                sim[i_x, :] = phase_data[i_x, :] * res.beta[0]
                filt[i_x, :] = mag_data[i_x, :] - est
                # estimate residuals
                residuals[i_x, :] = np.sign(mag_data[i_x, :] - est) * (
                    np.sum(res.delta**2, axis=0) + res.eps**2
                )
                # res.delta --> Array of estimated errors in input variables (same shape as i_x)
                delta[i_x, :] = np.sum(res.delta, axis=0)
                # res.eps --> Array of estimated errors in response variables (same shape as y)
                eps[i_x, :] = res.eps
                xshift[i_x, :] = np.sum(res.xplus, axis=0)
                estimate[i_x, :] = res.y  # res.xplus --> Array of x + delta

        # Save outputs
        self._results["sim"] = fname_presuffix(
            self.inputs.phase,
            prefix="sim_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img1 = nb.Nifti1Image(
            np.reshape(sim, out_shape),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img1, self._results["sim"])

        self._results["filt"] = fname_presuffix(
            self.inputs.phase,
            prefix="filt_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img2 = nb.Nifti1Image(
            np.reshape(filt, out_shape),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img2, self._results["filt"])

        self._results["residuals"] = fname_presuffix(
            self.inputs.phase,
            prefix="residuals_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img3 = nb.Nifti1Image(
            np.reshape(residuals, out_shape),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img3, self._results["residuals"])

        self._results["xres"] = fname_presuffix(
            self.inputs.phase,
            prefix="xres_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img4 = nb.Nifti1Image(
            np.reshape(delta, out_shape),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img4, self._results["xres"])

        self._results["yres"] = fname_presuffix(
            self.inputs.phase,
            prefix="yres_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img5 = nb.Nifti1Image(
            np.reshape(eps, out_shape),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img5, self._results["yres"])

        self._results["xplus"] = fname_presuffix(
            self.inputs.phase,
            prefix="xplus_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img6 = nb.Nifti1Image(
            np.reshape(xshift, out_shape),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img6, self._results["xplus"])

        # plot fit statistic info
        self._results["stdp"] = fname_presuffix(
            self.inputs.phase,
            prefix="stdp_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img7 = nb.Nifti1Image(
            np.reshape(stdp, out_shape[0:-1]),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img7, self._results["stdp"])

        self._results["stdm"] = fname_presuffix(
            self.inputs.phase,
            prefix="stdm_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img8 = nb.Nifti1Image(
            np.reshape(stdm, out_shape[0:-1]),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img8, self._results["stdm"])

        self._results["r2"] = fname_presuffix(
            self.inputs.phase,
            prefix="r2_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img9 = nb.Nifti1Image(
            np.reshape(r2, out_shape[0:-1]),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img9, self._results["r2"])

        self._results["estimate"] = fname_presuffix(
            self.inputs.phase,
            prefix="estimate_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        out_img10 = nb.Nifti1Image(
            np.reshape(estimate, out_shape),
            affine=mask_img.affine,
            header=mask_img.get_header(),
        )
        nb.save(out_img10, self._results["estimate"])

        return runtime


def _linear(beta, x):
    f = beta[0] * x + beta[1]
    return f
