if __name__ == '__main__':

    import exoplanet as xo
    import pymc3 as pm
    import pymc3_ext as pmx
    import aesara_theano_fallback.tensor as tt
    from celerite2.theano import terms, GaussianProcess
    import arviz as az
    import os
    from _utils import create_dirs

    import pandas as pd
    import numpy as np
    import lightkurve as lk

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    from math_utils import calc_veq, calc_prot
    from posterior_corner import plot_posteriors

    ################################################################################

    trace_end = "_trace.nc"
    diag_plot_end = "_diagnostics.png"
    post_plot_end = "_posteriors.png"
    trace_post_corner_end = "_trace_corner.png"

    ################################################################################

    class NewRotationTerm(terms.TermSum):

        def __init__(self, *, sigma, rho, period, f, **kwargs):
            self.sigma = tt.as_tensor_variable(sigma).astype("float64")
            self.rho = tt.as_tensor_variable(rho).astype("float64")
            self.period = tt.as_tensor_variable(period).astype("float64")
            self.f = tt.as_tensor_variable(f).astype("float64")

            w = 2 * np.pi / self.period
            a2 = 0.25 * self.f ** 2
            amp = self.sigma ** 2 / (1.0 + self.f + a2)

            super().__init__(
                terms.Matern32Term(sigma=tt.sqrt(amp), rho=np.sqrt(3) * rho),
                terms.ComplexTerm(a=self.f * amp, b=self.f * amp / (w * self.rho), c=1 / self.rho, d=w),
                terms.ComplexTerm(a=a2 * amp, b=a2 * amp / (2 * w * self.rho), c=1 / self.rho, d=2 * w),
            )

    ################################################################################

    def ready_data(y, yerr, data_type='lc'):
        truth = y - yerr
        yerr = np.abs(yerr)

        if data_type == 'lc' or data_type == 'lc_prime':
            mu = np.mean(y)
            y = (y / mu - 1) * 1e3
            yerr = yerr * 1e3 / mu
            truth = (truth / mu - 1) * 1e3

        return y, yerr, truth

    def calc_lsc(x, y, plot=False):

        results = xo.estimators.lomb_scargle_estimator(
            x, y, max_peaks=1, min_period=0.5, max_period=20.0, samples_per_peak=50
        )

        peak = results["peaks"][0]
        freq, power = results["periodogram"]

        if plot:

            fig, ax = plt.subplots(figsize=(12, 4))

            ax.plot(1 / freq, power, "k")
            ax.axvline(peak["period"], color="k", lw=4, alpha=0.3)

            ax.set_xlim((1 / freq).min(), (1 / freq).max())
            ax.set_yticks([])
            ax.set_xlabel("Period (days)")
            ax.set_ylabel("LSC Power")

            #plot_fn = 'Plots/variable_test_LSC_N{0:d}_i{1:.1f}_P{2:.1f}_t{3:d}.png'.format(N, inc, prot, len(t))
            #plt.savefig(plot_fn, format='png', bbox_inches='tight', dpi=400)

            plt.tight_layout()
            plt.show()

        return peak, freq, power

    def gp_rv(x, y, yerr, posteriors=False, trace_fn=None, kern_fn=None):

        peak, freq, power = calc_lsc(x, y)

        with pm.Model() as model:

            # The mean value of the time series
            mean = pm.Normal("mean", mu=0.0, sigma=1.0)

            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=1.0)

            # A term to describe the non-periodic variability
            sigma = pm.InverseGamma(
                "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0),
                testval=1.0)
            rho = pm.InverseGamma(
                "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 5.0))

            # Find functional form between rho and sigma,
            # i.e., log_rho(log_sigma) = m*log_sigma + b,
            # by fitting a line in log_rho v. log_sigma space and dividing
            # rho by that functional form (this will flatten rho in sigma
            # space, resulting in posteriors/values that look closer to the
            # MAP solution, where rho v sigma is flat?)

            # This forces sigma_rot > sigma
            log_sr_m_s = pm.Normal("log_sr_m_s", mu=0.0, sigma=3.0)

            sigma_rot = pm.Deterministic("sigma_rot", tt.exp(log_sr_m_s)+sigma)

            # The parameters of the RotationTerm kernel
            #sigma_rot = pm.InverseGamma(
            #    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 150.0),
            #    testval=51.0)
            log_period = pm.Normal("log_period", mu=np.log(peak["period"]), sigma=0.75)
            period = pm.Deterministic("period", tt.exp(log_period))
            log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
            log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=2.0)
            f = pm.Uniform("f", lower=0.1, upper=2.0)
            #f = 0.5

            # The parameters of the NewRotationTerm kernel
            #sigma_rot = pm.InverseGamma(
            #    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(0.5, 10.0),
            #    testval=2.0)
            #rho_rot = pm.InverseGamma(
            #    "rho_rot", **pmx.estimate_inverse_gamma_parameters(0.5, 10.0))
            #log_period = pm.Normal("log_period", mu=np.log(peak["period"]), sigma=3.0)
            #period = pm.Deterministic("period", tt.exp(log_period))
            ##f = pm.Uniform("f", lower=0.01, upper=10.0)
            #f = 0.5

            # Set up the Gaussian Process model
            kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1.0/3.0)
            kernel += terms.RotationTerm(
                sigma=sigma_rot,
                period=period,
                Q0=tt.exp(log_Q0),
                dQ=tt.exp(log_dQ),
                f=f,)

            #kernel += NewRotationTerm(
            #    sigma=sigma_rot,
            #    rho=rho_rot,
            #    period=period,
            #    f=f,)

            gp = GaussianProcess(
                kernel,
                t=x,
                diag=yerr ** 2 + tt.exp(2 * log_jitter),
                mean=mean,
                quiet=True,)

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            gp.marginal("gp", observed=y)

            # Compute the mean and sigma of model prediction for plotting purposes
            mu, variance = gp.predict(y, return_var=True)
            pm.Deterministic("pred", mu)
            pm.Deterministic("pred_sigma", tt.sqrt(variance))

            # evaluate kernel values at tau
            k_val = kernel.get_value(x - x[0])
            pm.Deterministic("k_val", k_val)

            # Optimize to find the maximum a posteriori parameters
            map_soln = model.test_point

            #map_soln = pmx.optimize(map_soln, vars=[f])
            #map_soln = pmx.optimize(map_soln, vars=[rho_rot])

            map_soln = pmx.optimize(map_soln, vars=[log_Q0])
            map_soln = pmx.optimize(map_soln, vars=[log_dQ])
            map_soln = pmx.optimize(map_soln, vars=[f])
            map_soln = pmx.optimize(map_soln, vars=[rho])
            map_soln = pmx.optimize(map_soln, vars=[f, rho, log_dQ, log_Q0])
            map_soln = pmx.optimize(map_soln, vars=[log_period])
            map_soln = pmx.optimize(map_soln, vars=[sigma])
            #map_soln = pmx.optimize(map_soln, vars=[sigma_rot])
            map_soln = pmx.optimize(map_soln, vars=[log_sr_m_s])
            map_soln = pmx.optimize(map_soln, vars=[log_period, sigma, log_sr_m_s])
            map_soln = pmx.optimize(map_soln, vars=[mean, log_jitter])

            #map_soln = pmx.optimize(map_soln, vars=[sigma, sigma_rot, log_sr_m_s])

            #vars = [sigma, sigma_rot, log_sr_m_s, rho, log_period, mean, log_jitter]
            #map_soln = pmx.optimize(map_soln, vars=vars)

            # Notes:
            # log_jitter and rho cause huge jumps in model
            # logp, often into positive value
            # The most problematic parameters are sigma/sigma_rot
            # (and, by extension, log_sr_m_s)

            #vars1 = [mean, sigma_rot, log_jitter, log_sr_m_s]
            #vars2 = [log_period, sigma, log_dQ, log_Q0, rho]

            #map_soln, info = pmx.optimize(map_soln, vars=vars2, return_info=True)
            #map_soln, info = pmx.optimize(map_soln, vars=vars1, return_info=True)

            #map_soln, info = pmx.optimize(map_soln, vars=vars1, return_info=True)
            #map_soln, info = pmx.optimize(map_soln, vars=vars2, return_info=True)

            #vars_combo_1 = [sigma_rot, log_jitter, f, rho, log_period]
            #vars_combo_1 = [sigma_rot, log_jitter, rho, log_period]
            #vars_combo_2 = [log_dQ, log_Q0, mean, sigma]

            # LC Light curve is SPOT ON with this one (for original N25)
            #vars_combo_1 = [log_sr_m_s, sigma_rot, sigma]
            #vars_combo_2 = [log_dQ, log_Q0, log_jitter, log_period, rho, mean]

            # Test
            #vars_combo_1 = [log_jitter, log_dQ, log_Q0, mean, sigma_rot]
            #vars_combo_2 = [log_period, rho]
            #vars_combo_3 = [sigma]
            #vars_combo_4 = [log_sr_m_s]

            # NewRotationTerm Kernel
            #vars_combo_1 = [sigma_rot, log_jitter, rho, log_period]
            #vars_combo_2 = [mean, sigma, rho_rot]

            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_1, return_info=True)
            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_2, return_info=True)
            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_3, return_info=True)
            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_4, return_info=True)
            #print("")
            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_1, return_info=True)
            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_2, return_info=True)
            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_3, return_info=True)
            #map_soln, info = pmx.optimize(start=map_soln, vars=vars_combo_4, return_info=True)

            #map_soln, info = pmx.optimize(map_soln, return_info=True)

            #print(-1*info["fun"])

            opt_params = map_soln.copy()
            del opt_params['pred']
            del opt_params['pred_sigma']
            del opt_params['k_val']

            opt_params = pd.DataFrame(opt_params, index=[0])
            opt_params.to_csv(kern_fn + "_rv.csv",
                              index=False)

            # New, optimized kernel
            opt_k = terms.SHOTerm(sigma=map_soln["sigma"], rho=map_soln["rho"], Q=1.0/3.0)
            opt_k += terms.RotationTerm(
                sigma=map_soln["sigma_rot"],
                period=tt.exp(map_soln["log_period"]),
                Q0=tt.exp(map_soln["log_Q0"]),
                dQ=tt.exp(map_soln["log_dQ"]),
                f=map_soln["f"],)

            #opt_k += NewRotationTerm(
            #    sigma=map_soln["sigma_rot"],
            #    rho=map_soln["rho_rot"],
            #    period=tt.exp(map_soln["log_period"]),
            #    #f=map_soln["f"],)
            #    f=map_soln["f"],)

            # New, optimized gp
            #opt_gp = GaussianProcess(
            #    opt_k,
            #    t=x,
            #    diag=yerr ** 2 + tt.exp(2 * map_soln["log_jitter"]),
            #    mean=map_soln["mean"],
            #    quiet=True,)

            if posteriors:
                posteriors_fn = trace_fn + '_rv' + trace_end

                if os.path.exists(posteriors_fn):
                    trace = az.from_netcdf(posteriors_fn)
                else:
                    trace = pmx.sample(
                        tune=1000,
                        draws=1000,
                        start=map_soln,
                        cores=2,
                        chains=2,
                        target_accept=0.9,
                        return_inferencedata=True,)

                    trace.to_netcdf(posteriors_fn)

                return map_soln, opt_k, trace
            else:
                return map_soln, opt_k

    def gp_lc(x, y, yerr, posteriors=False, trace_fn=None, kern_fn=None):

        peak, freq, power = calc_lsc(x, y)

        with pm.Model() as model:

            # The mean value of the time series
            mean = pm.Normal("mean", mu=0.0, sigma=1.0)

            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=1.0)

            # A term to describe the non-periodic variability
            sigma = pm.InverseGamma(
                "sigma", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0),
                testval=1.5)
            rho = pm.InverseGamma(
                "rho", **pmx.estimate_inverse_gamma_parameters(1.0, 20.0))

            #sigma_rot = pm.InverseGamma(
            #    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(0.5, 5.0),
            #    testval=3.0)
            log_period = pm.Normal("log_period", mu=np.log(peak["period"]), sigma=0.75)
            period = pm.Deterministic("period", tt.exp(log_period))
            log_Q0 = pm.HalfNormal("log_Q0", sigma=1.0)
            log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=1.0)
            f = pm.Uniform("f", lower=0.1, upper=2.0)

            # This forces sigma_rot > sigma
            log_sr_m_s = pm.Normal("log_sr_m_s", mu=0.0, sigma=1.0)
            sigma_rot = pm.Deterministic("sigma_rot", tt.exp(log_sr_m_s)+sigma)

            #log_sr_m_s = pm.Normal("log_sr_m_s", mu=tt.log(sigma_rot-sigma), sigma=5.0,
            #                       testval=np.log(1.5))

            kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1.0/3.0)
            kernel += terms.RotationTerm(
                sigma=sigma_rot,
                period=period,
                Q0=tt.exp(log_Q0),
                dQ=tt.exp(log_dQ),
                f=f,)

            gp = GaussianProcess(
                kernel,
                t=x,
                diag=yerr ** 2 + tt.exp(2 * log_jitter),
                mean=mean,
                quiet=True,)

            gp.marginal("gp", observed=y)

            # Compute the mean and sigma of model prediction for plotting purposes
            mu, variance = gp.predict(y, return_var=True)
            pm.Deterministic("pred", mu)
            pm.Deterministic("pred_sigma", tt.sqrt(variance))

            # evaluate kernel values at tau
            k_val = kernel.get_value(x - x[0])
            pm.Deterministic("k_val", k_val)

            map_soln = model.test_point

            #map_soln = pmx.optimize(map_soln, vars=[f])
            #map_soln = pmx.optimize(map_soln, vars=[rho_rot])

            map_soln = pmx.optimize(map_soln, vars=[log_Q0])
            map_soln = pmx.optimize(map_soln, vars=[log_dQ])
            map_soln = pmx.optimize(map_soln, vars=[f])
            map_soln = pmx.optimize(map_soln, vars=[rho])
            map_soln = pmx.optimize(map_soln, vars=[f, rho, log_dQ, log_Q0])
            map_soln = pmx.optimize(map_soln, vars=[log_period])
            map_soln = pmx.optimize(map_soln, vars=[sigma])
            #map_soln = pmx.optimize(map_soln, vars=[sigma_rot])
            map_soln = pmx.optimize(map_soln, vars=[log_sr_m_s])
            map_soln = pmx.optimize(map_soln, vars=[log_period, sigma, log_sr_m_s])
            map_soln = pmx.optimize(map_soln, vars=[mean, log_jitter])

            opt_params = map_soln.copy()
            del opt_params['pred']
            del opt_params['pred_sigma']
            del opt_params['k_val']

            opt_params = pd.DataFrame(opt_params, index=[0])
            opt_params.to_csv(kern_fn + "_lc.csv",
                              index=False)

            # New, optimized kernel
            opt_k = terms.SHOTerm(sigma=map_soln["sigma"], rho=map_soln["rho"], Q=1.0/3.0)
            opt_k += terms.RotationTerm(
                sigma=map_soln["sigma_rot"],
                period=tt.exp(map_soln["log_period"]),
                Q0=tt.exp(map_soln["log_Q0"]),
                dQ=tt.exp(map_soln["log_dQ"]),
                f=map_soln["f"],)

            if posteriors:
                posteriors_fn = trace_fn + '_lc' + trace_end

                if os.path.exists(posteriors_fn):
                    trace = az.from_netcdf(posteriors_fn)
                else:
                    trace = pmx.sample(
                        tune=1000,
                        draws=1000,
                        start=map_soln,
                        cores=2,
                        chains=2,
                        target_accept=0.9,
                        return_inferencedata=True,)

                    trace.to_netcdf(posteriors_fn)

                return map_soln, opt_k, trace
            else:
                return map_soln, opt_k

    def gp_lc_prime(x, y, yerr, posteriors=False, trace_fn=None, kern_fn=None):

        peak, freq, power = calc_lsc(x, y)

        with pm.Model() as model:

            # The mean value of the time series
            mean = pm.Normal("mean", mu=0.0, sigma=1.0)

            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=1.0)

            # A term to describe the non-periodic variability
            sigma = pm.InverseGamma(
                "sigma", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0))
            rho = pm.InverseGamma(
                "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 5.0))

            # This forces sigma_rot > sigma
            log_sr_m_s = pm.Normal("log_sr_m_s", mu=0.0, sigma=1.0)
            sigma_rot = pm.Deterministic("sigma_rot", tt.exp(log_sr_m_s)+sigma)

            #sigma_rot = pm.InverseGamma(
            #    "sigma_rot", **pmx.estimate_inverse_gamma_parameters(0.5, 150.0),
            #    testval=51.0)
            log_period = pm.Normal("log_period", mu=np.log(peak["period"]), sigma=0.75)
            period = pm.Deterministic("period", tt.exp(log_period))
            log_Q0 = pm.HalfNormal("log_Q0", sigma=1.0)
            log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=1.0)
            f = pm.Uniform("f", lower=0.1, upper=2.0)

            #log_sr_m_s = pm.Normal("log_sr_m_s", mu=tt.log(sigma_rot-sigma), sigma=3.0,
            #                       testval=np.log(51.0 - 1.5))

            kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1.0/3.0)
            kernel += terms.RotationTerm(
                sigma=sigma_rot,
                period=period,
                Q0=tt.exp(log_Q0),
                dQ=tt.exp(log_dQ),
                f=f,)

            gp = GaussianProcess(
                kernel,
                t=x,
                diag=yerr ** 2 + tt.exp(2 * log_jitter),
                mean=mean,
                quiet=True,)

            gp.marginal("gp", observed=y)

            # Compute the mean and sigma of model prediction for plotting purposes
            mu, variance = gp.predict(y, return_var=True)
            pm.Deterministic("pred", mu)
            pm.Deterministic("pred_sigma", tt.sqrt(variance))

            # evaluate kernel values at tau
            k_val = kernel.get_value(x - x[0])
            pm.Deterministic("k_val", k_val)

            map_soln = model.test_point

            #map_soln = pmx.optimize(map_soln, vars=[f])
            #map_soln = pmx.optimize(map_soln, vars=[rho_rot])

            map_soln = pmx.optimize(map_soln, vars=[log_Q0])
            map_soln = pmx.optimize(map_soln, vars=[log_dQ])
            map_soln = pmx.optimize(map_soln, vars=[f])
            map_soln = pmx.optimize(map_soln, vars=[rho])
            map_soln = pmx.optimize(map_soln, vars=[f, rho, log_dQ, log_Q0])
            map_soln = pmx.optimize(map_soln, vars=[log_period])
            map_soln = pmx.optimize(map_soln, vars=[sigma])
            #map_soln = pmx.optimize(map_soln, vars=[sigma_rot])
            map_soln = pmx.optimize(map_soln, vars=[log_sr_m_s])
            map_soln = pmx.optimize(map_soln, vars=[log_period, sigma, log_sr_m_s])
            map_soln = pmx.optimize(map_soln, vars=[mean, log_jitter])

            opt_params = map_soln.copy()
            del opt_params['pred']
            del opt_params['pred_sigma']
            del opt_params['k_val']

            opt_params = pd.DataFrame(opt_params, index=[0])
            opt_params.to_csv(kern_fn + "_lc_prime.csv",
                              index=False)

            # New, optimized kernel
            opt_k = terms.SHOTerm(sigma=map_soln["sigma"], rho=map_soln["rho"], Q=1.0/3.0)
            opt_k += terms.RotationTerm(
                sigma=map_soln["sigma_rot"],
                period=tt.exp(map_soln["log_period"]),
                Q0=tt.exp(map_soln["log_Q0"]),
                dQ=tt.exp(map_soln["log_dQ"]),
                f=map_soln["f"],)

            if posteriors:
                posteriors_fn = trace_fn + '_lc_prime' + trace_end

                if os.path.exists(posteriors_fn):
                    trace = az.from_netcdf(posteriors_fn)
                else:
                    trace = pmx.sample(
                        tune=1000,
                        draws=1000,
                        start=map_soln,
                        cores=2,
                        chains=2,
                        target_accept=0.9,
                        return_inferencedata=True,)

                    trace.to_netcdf(posteriors_fn)

                return map_soln, opt_k, trace
            else:
                return map_soln, opt_k

    def plot_post_kernels(t, tau, trace, ax_data, ax_kernels, color, chains=2, ndraws=1000, max_plot=50):

        plot_per_chain = int(max_plot / chains)

        pred, k_val = [], []
        for c in range(chains):
            post = trace.posterior

            # Plot the ith posterior draw
            draws = np.random.randint(low=0, high=ndraws, size=plot_per_chain)
            for i in draws:
                #ax_data.plot(t, post['pred'][c][i], color=color,
                #           alpha=0.1, zorder=6, lw=4.0, rasterized=True)

                pred.append(np.array(post['pred'][c][i]))

                k_i = post["k_val"][c][i]#.eval()
                k_i /= k_i[0]

                k_val.append(np.array(k_i))

                #ax_kernels[0].plot(tau, k_i, color=color, rasterized=True,
                #                   lw=2.0, ls='-', alpha=0.1)
                #ax_kernels[1].plot(tau, k_i, color=color, rasterized=True,
                #                   lw=2.0, ls='-', alpha=0.1)

        t = np.tile(t, (len(pred), 1))
        tau = np.tile(tau, (len(k_val), 1))

        pred, k_val = np.array(pred), np.array(k_val)

        ax_data.plot(t.T, pred.T, color=color,
                   alpha=0.1, zorder=6, lw=4.0, rasterized=True)

        ax_kernels[0].plot(tau.T, k_val.T, color=color, rasterized=True,
                           lw=2.0, ls='-', alpha=0.1)
        ax_kernels[1].plot(tau.T, k_val.T, color=color, rasterized=True,
                           lw=2.0, ls='-', alpha=0.1)

    def acf_and_kernel(x, y, opt_k):

        tau = x - x[0]
        acor = xo.estimators.autocorr_function(y)
        k_val = opt_k.get_value(tau).eval()
        k_val /= k_val[0] # normalize since acf is normalized

        return tau, acor, k_val

    def plot_single_model(x, y, yerr, data_type='lc'):
        """
        07.10.2021 : Outdated, will give errors
        """

        y, yerr, truth = ready_data(y, yerr, data_type=data_type)

        map_soln, opt_k = gp_rv(x, y, yerr)
        tau, acor, k_val = acf_and_kernel(x, y, opt_k)

        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(18, 8),
                               gridspec_kw={'hspace':0.25})

        ax[0].plot(x, truth, "k", lw=3.0, alpha=0.5, label='Truth', zorder=10)

        ax[0].scatter(x, y, c='black', marker='o', s=2.5, label='Data', zorder=-1)

        ax[0].plot(x, map_soln["pred"], color="C0",
                    alpha=0.65, zorder=6,
                    label=r'$\mu_\mathrm{GP} \pm 3\sigma_\mathrm{GP}$')
        ax[0].fill_between(x, map_soln["pred"]-3.*map_soln["pred_sigma"],
                        map_soln["pred"]+3.*map_soln["pred_sigma"],
                        color="C0", alpha=0.4, zorder=6)

        ax[0].set_xlim(x.min(), x.max())
        ax[0].set_xlabel(r"Time (days)", fontsize=24)

        if data_type == 'lc':
            ax[0].set_ylabel(r"Relative Flux (ppt)", fontsize=22)
        elif data_type == 'rv':
            ax[0].set_ylabel(r"Radial Velocity (m s$^{-1}$)", fontsize=22)


        ax[0].legend(loc='lower left', fontsize=20)

        ####

        ax[1].plot(tau, acor, color='#F58D2F', lw=2.5, ls='--', label='ACF')
        ax[1].plot(tau, k_val, color='#5198E6', lw=2.5, ls='-', label='GP Kernel')

        ax[1].set_xlim(tau.min(), tau.max())
        ax[1].set_xlabel(r"$\tau$ (days)", fontsize=24)

        if data_type == 'lc':
            ax[1].set_ylabel(r"ACF or Kernel (ppt$^2$)", fontsize=22)
        elif data_type == 'rv':
            ax[1].set_ylabel(r"ACF or Kernel (m$^2$ s$^{-2})", fontsize=22)

        ax[1].legend(fontsize=20)

        #prot_text   = r'$v_\mathrm{{eq}} = {0:.1f}$ km s$^{{-1}}$, '.format(veq)
        #nspots_text = r'$N_\mathrm{{spots}} = {0}$, '.format(N)
        #inclin_text = r'$i_{{\star}} = {0:.1f} \:$ $\ocirc$, '.format(inc)
        #obli_text   = r'$\Psi_{{\star}} = {0:.1f} \:$ $\ocirc$, '.format(obl)
        #sp_deg_text = r'max Y$_{{l,m}} = {0} \:$ $\ocirc$'.format(20)
        #star_title  = prot_text + nspots_text + inclin_text + obli_text + sp_deg_text

        #plt.suptitle(star_title, size=24)

        if data_type == 'lc':
            plot_fn = 'Plots/lc_model_acf_kernel_N{0:d}_P{1:.1f}'.format(N, prot)
        elif data_type == 'rv':
            plot_fn = 'Plots/rv_model_acf_kernel_N{0:d}_P{1:.1f}'.format(N, prot)

        #plt.savefig(plot_fn+'.png', format='png', bbox_inches='tight', dpi=400)
        #plt.savefig(plot_fn+'.pdf', format='pdf', bbox_inches='tight', dpi=400)
        plt.tight_layout()
        plt.show()

    def plot_double_model(t, f, f_err, t_rv, rv, rv_err, posteriors=False,
                          kern_fn=None, trace_fn=None, plot_fn=None):

        # index goes row i and col j
        fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(50, 30),
                               gridspec_kw={'hspace':0.25, 'wspace':0.15,
                                            'height_ratios':[1.0, 1.0, 0.1, 1.0]})

        # Change the axis linewidth for all axes
        # and also the tick sizes to fit the size of figure
        for a in ax:
            for ai in a:
                ai.tick_params(axis='both', which='major', width=3.0, size=12, labelsize=36)
                ai.tick_params(axis='both', which='minor', width=2.75, size=7, labelsize=36)

                for s in ['top','bottom','left','right']:
                    ai.spines[s].set_linewidth(4)

        # Remove the 3rd row, which is used just to make extra space
        ax[2, 0].remove()
        ax[2, 1].remove()

        # label individual axes
        lc_ax = ax[0, 0] # first row, first column
        rv_ax = ax[1, 0] # second row, first column
        lc_acf_k_ax = ax[0, 1] # first row, second column
        rv_acf_k_ax = ax[1, 1] # second row, second column

        acf_v_acf_ax = ax[3, 0] # last row, first column
        k_v_k_ax = ax[3, 1] # last row, second column

        # Do all the labels
        lc_ax.set_xlabel(r"Time (days)", fontsize=48)
        rv_ax.set_xlabel(r"Time (days)", fontsize=48)
        lc_acf_k_ax.set_xlabel(r"$\tau$ (days)", fontsize=48)
        rv_acf_k_ax.set_xlabel(r"$\tau$ (days)", fontsize=48)
        acf_v_acf_ax.set_xlabel(r"$\tau$ (days)", fontsize=48)
        k_v_k_ax.set_xlabel(r"$\tau$ (days)", fontsize=48)

        lc_ax.set_ylabel(r"Relative Flux (ppt)", fontsize=48)
        rv_ax.set_ylabel(r"Radial Velocity (m s$^{-1}$)", fontsize=48)
        lc_acf_k_ax.set_ylabel(r"ACF$_\mathrm{phot}$ or GP$_\mathrm{phot}$ (ppt$^2$)", fontsize=48)
        rv_acf_k_ax.set_ylabel(r"ACF$_\mathrm{rv}$ or GP$_\mathrm{rv}$ (m$^2$ s$^{-2}$)", fontsize=48)
        acf_v_acf_ax.set_ylabel(r"ACF (ppt$^2$ or m$^2$ s$^{-2}$)", fontsize=48)
        k_v_k_ax.set_ylabel(r"Kernel (ppt$^2$ or m$^2$ s$^{-2}$)", fontsize=48)

        # Do the titles
        lc_ax.set_title(r"Data and MAP GP", fontsize=60)
        lc_acf_k_ax.set_title(r"ACF vs. Kernel", fontsize=60)
        acf_v_acf_ax.set_title(r"ACF$_\mathrm{phot}$ vs. ACF$_\mathrm{rv}$", fontsize=60)
        k_v_k_ax.set_title(r"Kernel$_\mathrm{phot}$ vs. Kernel$_\mathrm{rv}$", fontsize=60)

        # Separate the bottom panels
        line = plt.Line2D((.05,.95),(.37, .37), color="grey", linewidth=5.0, ls='-')
        fig.add_artist(line)

        # Set colors
        lc_c = '#5198E6'
        rv_c = '#F58D2F'

        # Grab all data
        f, f_err, f_truth = ready_data(f, f_err, data_type='lc')
        rv, rv_err, rv_truth = ready_data(rv, rv_err, data_type='rv')

        if posteriors:
            f_map_soln, f_opt_k, f_trace = gp_lc(t, f, f_err, kern_fn=kern_fn,
                                                 posteriors=posteriors, trace_fn=trace_fn)

        else:
            f_map_soln, f_opt_k = gp_lc(t, f, f_err, kern_fn=kern_fn)

        f_tau, f_acor, f_k_val = acf_and_kernel(t, f, f_opt_k)

        if posteriors:
            rv_map_soln, rv_opt_k, rv_trace = gp_rv(t_rv, rv, rv_err, kern_fn=kern_fn,
                                                    posteriors=posteriors, trace_fn=trace_fn)
        else:
            rv_map_soln, rv_opt_k = gp_rv(t_rv, rv, rv_err, kern_fn=kern_fn)

        rv_tau, rv_acor, rv_k_val = acf_and_kernel(t_rv, rv, rv_opt_k)

        # Plot all data
        # Light curve
        lc_ax.plot(t, f_truth, "k", alpha=0.5, lw=3.5, label='Truth', zorder=10)

        lc_ax.scatter(t, f, c='black', marker='o', s=10.0, label='Data', zorder=-1)

        if posteriors:
            plot_post_kernels(t, f_tau, f_trace, lc_ax, [lc_acf_k_ax, k_v_k_ax], lc_c, max_plot=100)
        else:
            lc_ax.plot(t, f_map_soln["pred"], color=lc_c,
                       alpha=0.65, zorder=6, lw=3.5,
                       label=r'$\mu_\mathrm{GP} \pm 3\sigma_\mathrm{GP}$')
            lc_ax.fill_between(t, f_map_soln["pred"]-3.*f_map_soln["pred_sigma"],
                            f_map_soln["pred"]+3.*f_map_soln["pred_sigma"],
                            color=lc_c, alpha=0.4, zorder=6)
            lc_acf_k_ax.plot(f_tau, f_k_val, color=lc_c, lw=4.5, ls='-', label='GP$_\mathrm{phot}$ Kernel')

        lc_acf_k_ax.plot(f_tau, f_acor, color='black', lw=4.0, ls='--', label='ACF$_\mathrm{phot}$')

        # RV curve
        rv_ax.plot(t_rv, rv_truth, "k", lw=3.5, alpha=0.5, label='Truth', zorder=10)

        rv_ax.scatter(t_rv, rv, c='black', marker='o', s=10.0, label='Data', zorder=-1)

        if posteriors:
            plot_post_kernels(t_rv, rv_tau, rv_trace, rv_ax, [rv_acf_k_ax, k_v_k_ax], rv_c, max_plot=100)
        else:
            rv_ax.plot(t_rv, rv_map_soln["pred"], color=rv_c, alpha=0.65, lw=3.5,
                       label=r'$\mu_\mathrm{GP} \pm 3\sigma_\mathrm{GP}$', zorder=6)
            rv_ax.fill_between(t_rv, rv_map_soln["pred"]-3.*rv_map_soln["pred_sigma"],
                            rv_map_soln["pred"]+3.*rv_map_soln["pred_sigma"],
                            color=rv_c, alpha=0.4, zorder=6)
            rv_acf_k_ax.plot(rv_tau, rv_k_val, color=rv_c, lw=4.5, ls='-', label='GP$_\mathrm{rv}$ Kernel')

        rv_acf_k_ax.plot(rv_tau, rv_acor, color='black', lw=4.0, ls='--', label='ACF$_\mathrm{rv}$')

        # ACFs and Kernels
        acf_v_acf_ax.plot(f_tau, f_acor, color=lc_c, lw=4.0, ls='--', label='ACF$_\mathrm{phot}$')
        acf_v_acf_ax.plot(rv_tau, rv_acor, color=rv_c, lw=4.0, ls='--', label='ACF$_\mathrm{rv}$')

        if not posteriors:
            k_v_k_ax.plot(f_tau, f_k_val, color=lc_c, lw=4.0, ls='-', label='GP$_\mathrm{phot}$ Kernel')
            k_v_k_ax.plot(rv_tau, rv_k_val, color=rv_c, lw=4.0, ls='-', label='GP$_\mathrm{rv}$ Kernel')

        # Change limits of axes
        lc_ax.set_xlim(t.min(), t.max())
        rv_ax.set_xlim(t.min(), t.max())

        lc_acf_k_ax.set_xlim(f_tau.min(), f_tau.max())
        rv_acf_k_ax.set_xlim(rv_tau.min(), rv_tau.max())

        acf_v_acf_ax.set_xlim(min(min(f_tau), min(rv_tau)), max(max(f_tau), max(rv_tau)))
        k_v_k_ax.set_xlim(min(min(f_tau), min(rv_tau)), max(max(f_tau), max(rv_tau)))

        # Set legends
        lc_ax.legend(fontsize=36)
        rv_ax.legend(fontsize=36)
        lc_acf_k_ax.legend(fontsize=36)
        rv_acf_k_ax.legend(fontsize=36)
        acf_v_acf_ax.legend(fontsize=36)

        if not posteriors:
            k_v_k_ax.legend(fontsize=36)

        # Total figure title
        prot_text   = r'$P_\mathrm{{rot}} = {0:.1f}$ day, '.format(prot)
        nspots_text = r'$N_\mathrm{{spots}} = {0}$, '.format(N)
        inclin_text = r'$i_{{\star}} = {0:.1f} \:$ $\ocirc$, '.format(inc)
        obli_text   = r'$\Psi_{{\star}} = {0:.1f} \:$ $\ocirc$, '.format(obl)
        sp_deg_text = r'max Y$_{{l,m}} = {0} \:$ $\ocirc$'.format(20)
        star_title  = prot_text + nspots_text + inclin_text + obli_text + sp_deg_text

        plt.suptitle(star_title, size=82)

        plt.savefig(plot_fn, format='png', bbox_inches='tight', dpi=400)

        plt.close()

    def get_trace(fn):
        trace = az.from_netcdf(fn)
        return trace

    def print_soln_params(t, f, f_err):
        """
        07.10.2021 : Outdated and will return errors
        """

        f, f_err, f_truth = ready_data(f, f_err, data_type='lc')
        map_soln, opt_k = gp_rv(t, f, f_err)

        sigma = map_soln["sigma"]
        rho = map_soln["rho"]
        sigma_rot = map_soln["sigma_rot"]
        period = map_soln["period"]
        logQ0 = map_soln["log_Q0"]
        logdQ = map_soln["log_dQ"]
        f = map_soln["f"]
        log_jit = map_soln["log_jitter"]

        print("")
        print("log_jitter = {0:.3f}".format(log_jit))
        print("log_dQ = {0:.3f}".format(logdQ))
        print("sigma = {0:.3f}".format(sigma))
        print("rho = {0:.3f}".format(rho))
        print("sigma_rot = {0:.3f}".format(sigma_rot))
        print("period = {0:.3f}".format(period))
        print("log_Q0 = {0:.3f}".format(logQ0))
        print("f = {0:.3f}".format(f))

    def plot_period_dist(x, y, yerr):
        """
        x : ndarray of all x values
        y : ndarray of all y values
        """

        # Set colors
        lc_c = '#5198E6'
        lc_prime_c = '#8000DC'
        rv_c = '#F58D2F'

        d = ['lc', 'lc_prime', 'rv']
        c = [lc_c, lc_prime_c, rv_c]
        l = [r"$F$", r"$F^\prime$", r"RV"]

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(18, 4),
                               gridspec_kw={'wspace':0.1, 'width_ratios':[1.0, 0.5]})

        for i in range(len(x)):
            peak, freq, power = calc_lsc(x[i], y[i])

            yi, yerri, truth = ready_data(y[i], yerr[i], data_type=d[i])

            if i == 2:
                map_soln, opt_k, trace = gp_rv(x[i], yi, yerri, posteriors=True)
            elif i == 0:
                map_soln, opt_k, trace = gp_lc(x[i], yi, yerri, posteriors=True)
            elif i == 1:
                map_soln, opt_k, trace = gp_lc_prime(x[i], yi, yerri, posteriors=True)

            ax[0].plot(1 / freq, power/np.max(power), color=c[i], lw=2.0, label=l[i])
            ax[0].axvline(peak["period"], color=c[i], lw=5.0, alpha=0.25)


            names = list(trace.posterior.data_vars)
            names.remove("pred")
            names.remove("pred_sigma")
            names.remove("k_val")

            summary = az.summary(trace, var_names=names)
            per_post_mean = np.exp(summary['mean']["log_period"])
            periods = trace.posterior["log_period"].values.flatten()

            ax[1].axvline(per_post_mean, color=c[i], lw=1.0, ls='-')
            ax[1].axvline(float(np.exp(map_soln["log_period"])), color=c[i], lw=1.0, ls='--')
            ax[1].hist(np.exp(periods), 50, density=True, histtype='stepfilled',
                       facecolor=c[i], alpha=0.35)

            #print(d[i])
            #print(peak["period"])
            #print(per_post_mean)
            #print(float(np.exp(map_soln["log_period"])))
            #print("")

        ax[0].set_xlabel("Period (days)")
        ax[0].set_ylabel("Normalized LSC Power", size=20)
        ax[0].set_xlim([0.5, 20.0])
        ax[0].set_ylim(bottom=0.0)
        ax[0].set_yticks([])

        ax[1].set_xlabel("Period (days)")
        ax[1].set_xlim([0.5, 20.0])
        ax[1].set_ylim(bottom=0.0)
        ax[1].set_yticks([])

        ax[0].legend(loc='upper right', fontsize=20)

        plt.savefig('period_lsc.png', bbox_inches='tight')
        #plt.tight_layout()
        #plt.show()

    #N, inc, obl, prot, veq = 25, 85.0, 5.5, 7.2, calc_veq(7.2)
    #N, inc, obl, prot, veq = 5, 78.0, 4.1, 4.8, calc_veq(4.8)
    #N, inc, obl, prot, veq = 15, 87.2, 4.5, 5.8, calc_veq(5.8)
    #N, inc, obl, prot, veq = 37, 43.2, 3.2, 12.2, calc_veq(12.2)
    #N, inc, obl, prot, veq = 12, 80.4, 6.2, 3.1, calc_veq(3.1)
    # just like above but with spots pushed to poles
    #N, inc, obl, prot, veq = 12, 80.3, 6.3, 3.1, calc_veq(3.1)
    #N, inc, obl, prot, veq = 21, 72.1, 9.1, calc_prot(22.5), 22.5
    # just like above but with spots pushed to poles
    #N, inc, obl, prot, veq = 21, 72.0, 9.2, calc_prot(22.5), 22.5
    #N, inc, obl, prot, veq = 21, 71.9, 9.3, calc_prot(22.5), 22.5

    #t = np.zeros(960) # 20d, kepler_cadence
    #t = np.zeros(1440) # 30d, kepler_cadence

    #base_name = 'NewRotationTerm_'
    #base_name = 'RotationTerm_'
    #base_fn = "_N{0:d}_P{1:.1f}".format(N, prot)
    #spots_fn = "_N{0:d}_i{1:.1f}_P{2:.1f}_t{3:d}".format(N, inc, prot, len(t))

    ############################################################################

    #N, inc, obl, prot, veq = 25, 88.0, 4.0, 8.0, calc_veq(8.0)
    N, inc, obl, prot, veq = 50, 86.5, 6.0, 11.0, calc_veq(11.0)
    N_ori = 3

    taus = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
            0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

    def vary_spot_param(vars, spot_param="Decay Rate", get_posteriors=False, N_ori=N_ori):

        if spot_param == "Decay Rate":
            var_name = "tau_"
        if spot_param == "Spot Radius":
            var_name = "rad_"
        if spot_param == "Spot Contrast":
            var_name = "con_"
        if spot_param == "Spot Radius and Contrast":
            var_name = ""

        paths = create_dirs(N_ori, N, inc, obl, prot, veq)
        spot_param = spot_param + "/"

        var_paths = [i+spot_param for i in paths]

        var_dir = var_paths[0]
        op_kern_param_path = var_paths[1]
        diag_plot_path = var_paths[2]
        trace_path = var_paths[3]
        trace_post_path = var_paths[4]

        for p in var_paths:
            if not os.path.isdir(p):
                os.mkdir(p)

        for var in vars:

            var_path = var_dir + "/"+var_name+"{0:.3f}".format(var)

            t = np.load(var_path + "_t.npy")
            f = np.load(var_path + "_f.npy")
            f_err = np.load(var_path + "_ferr.npy")

            rv = np.load(var_path + "_rv.npy")
            rv_err = np.load(var_path + "_rverr.npy")

            #f_prime = np.load(var_path + "_f_prime.npy")
            #f_prime_err = np.load(var_path + "_f_prime_err.npy")

            # Outdated!!
            #plot_period_dist([t, t, t], [f, f_prime, rv],
            #                 [f_err, f_prime_err, rv_err])

            print("")
            print(spot_param[:-1] + " = {0:.3f}".format(var))

            kern_path = op_kern_param_path + var_name + "{0:.3f}".format(var)
            trace_path = trace_path + var_name + "{0:.3f}".format(var)

            if get_posteriors:
                plot_fn = diag_plot_path + var_name + "{0:.3f}".format(var) + post_plot_end
                plot_double_model(t, f, f_err, t, rv, rv_err, posteriors=True,
                                  plot_fn=plot_fn, kern_fn=kern_path,
                                  trace_fn=trace_path)
            else:
                plot_fn = diag_plot_path + var_name + "{0:.3f}".format(var) + diag_plot_end
                plot_double_model(t, f, f_err, t, rv, rv_err, posteriors=False,
                                  plot_fn=plot_fn, kern_fn=kern_path)

        #f_prime, f_prime_err, truth = ready_data(f_prime, f_prime_err, data_type='lc_prime')
        #gp_lc_prime(t, f_prime, np.abs(f_prime_err))

        if get_posteriors:
            posteriors_fn = trace_path + var_name + "{0:.3f}".format(var) + '_lc' + trace_end
            trace = get_trace(posteriors_fn)
            corner_plot_fn = trace_post_path + var_name + "{0:.3f}".format(var) + '_lc' + trace_post_corner_end
            plot_posteriors(trace, type='lc', fn=corner_plot_fn, use_corner=True)

            posteriors_fn = trace_path + var_name + "{0:.3f}".format(var) + '_rv' + trace_end
            trace = get_trace(posteriors_fn)
            corner_plot_fn = trace_post_path + var_name + "{0:.3f}".format(var) + '_rv' + trace_post_corner_end
            plot_posteriors(trace, type='rv', fn=corner_plot_fn, use_corner=True)

            # out of date
            #posteriors_fn = 'Traces/' + base_name + '_lc_prime' + trace_end
            #trace = get_trace(xposteriors_fn)
            #corner_plot_fn = 'Plots/Trace Posteriors/' + base_name + '_lc_prime' + trace_post_corner_end
            #plot_posteriors(trace, type='lc', fn=corner_plot_fn, use_corner=True)

    for n in [1]:
        vary_spot_param(taus, N_ori=n)

    """
    rads = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.7, 1.8, 1.9, 2.0]

    #rads = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    #        1.6, 1.7, 1.8, 1.9, 2.0]

    cons = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.5,
            1.6, 1.7, 1.8, 1.9, 2.0]

    for var in cons:
        base_fn = 'N{0:d}_inc{1:.1f}_obl{2:.1f}_P{3:.1f}'.format(N, inc, obl, prot)
        #base_name = "Spot Radius/" + base_fn + "/radius_{0:.2f}".format(rad)
        #base_name = "Spot Contrast/" + base_fn + "/contrast_{0:.2f}".format(var)
        base_name = "Spot Radius and Contrast/" + base_fn + "/{0:.2f}".format(var)

        t = np.load("Simulated Data/" + base_name + "_t.npy")
        f = np.load("Simulated Data/" + base_name + "_f.npy")
        f_err = np.load("Simulated Data/" + base_name + "_ferr.npy")

        rv = np.load("Simulated Data/" + base_name + "_rv.npy")
        rv_err = np.load("Simulated Data/" + base_name + "_rverr.npy")

        f_prime = np.load("Simulated Data/" + base_name + "_f_prime.npy")
        f_prime_err = np.load("Simulated Data/" + base_name + "_f_prime_err.npy")

        plot_double_model(t, f, f_err, t, rv, rv_err, posteriors=False)

        f_prime, f_prime_err, truth = ready_data(f_prime, f_prime_err, data_type='lc_prime')
        gp_lc_prime(t, f_prime, np.abs(f_prime_err))
    """
    ############################################################################

    """fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t, 1.0 - (f - f_err), lw=2.0, color='black', label='$1 - F$')
    ax.plot(t, f_prime, color='blue', lw=2.0, label='$F^\prime$')

    ax.set_xlabel(r"t (days)", fontsize=20)

    ax.legend(fontsize=24)

    ax.set_xlim([0.0, 20.0])

    plt.tight_layout()
    plt.show()"""

    #f, f_err, f_truth = ready_data(f, f_err, data_type='lc')
    #f_map_soln, f_opt_k, f_trace = gp_rv(t, f, f_err, posteriors=True)

    #print(f_trace.posterior["pred"][0][0])
    #print(f_trace.posterior["sigma_rot"][0])
