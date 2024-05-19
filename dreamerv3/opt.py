import jax
import jax.numpy as jnp
from jax.numpy import float32 as f32, int32 as i32
import numpy as np
import optax
from jax.tree_util import tree_map as treemap
import re
from jax.experimental import checkify
import collections
import contextlib
import equinox as eqx

COMPUTE_DTYPE = f32


def check(predicate, message, **kwargs):
    return False


def sg(x):
    return treemap(jax.lax.stop_gradient, x)


def parallel():
    try:
        jax.lax.axis_index('i')
        return True
    except NameError:
        return False


class Optimizer:
    # Normalization
    scaler: str = 'adam'
    eps: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.999

    # Learning rate
    warmup: int = 1000
    anneal: int = 0
    schedule: str = 'constant'

    # Regularization
    wd: float = 0.0
    wd_pattern: str = r'/kernel$'

    # Clipping
    pmin: float = 1e-3
    globclip: float = 0.0
    agc: float = 0.0

    # Smoothing
    momentum: bool = False
    nesterov: bool = False

    # Metrics
    details: bool = False

    def __init__(self, lr):
        self.lr = lr
        chain = []

        if self.globclip:
            chain.append(optax.clip_by_global_norm(self.globclip))
        if self.agc:
            chain.append(scale_by_agc(self.agc, self.pmin))

        if self.scaler == 'adam':
            chain.append(optax.scale_by_adam(self.beta1, self.beta2, self.eps))
        elif self.scaler == 'rms':
            chain.append(scale_by_rms(self.beta2, self.eps))
        else:
            raise NotImplementedError(self.scaler)

        if self.momentum:
            chain.append(scale_by_momentum(self.beta1, self.nesterov))

        if self.wd:
            assert not self.wd_pattern[0].isnumeric(), self.wd_pattern
            pattern = re.compile(self.wd_pattern)
            wdmaskfn = lambda params: {k: bool(pattern.search(k)) for k in params}
            chain.append(optax.add_decayed_weights(self.wd, wdmaskfn))

        if isinstance(self.lr, dict):
            chain.append(scale_by_groups({pfx: -lr for pfx, lr in self.lr.items()}))
        else:
            chain.append(optax.scale(-self.lr))

        self.chain = optax.chain(*chain)
        self.step = jnp.array(0, i32)
        self.scaling = (COMPUTE_DTYPE == jnp.float16)
        if self.scaling:
            self.chain = optax.apply_if_finite(
                self.chain, max_consecutive_errors=1000)
            self.grad_scale = jnp.array(1e4, f32)
            self.good_steps = jnp.array(0, i32)
        self.once = True
        self.opt_state = self.chain.init(eqx.filter(model, eqx.is_array))

    def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
        def wrapped(*args, **kwargs):
            outs = lossfn(*args, **kwargs)
            loss, aux = outs if has_aux else (outs, None)
            assert loss.dtype == f32, loss.dtype
            assert loss.shape == (), loss.dtype
            if self.scaling:
                loss *= sg(self.grad_scale)
            return loss, aux

        metrics = {}
        (loss, aux), grads = eqx.filter_value_and_grad(wrapped, has_aux=True)(modules, *args, **kwargs)

        if parallel():
            grads = treemap(lambda x: jax.lax.pmean(x, 'i'), grads)
        if self.scaling:
            invscale = 1.0 / self.grad_scale.read()
            grads = treemap(lambda x: x * invscale, grads)

        updates, opt_state = self.chain.update(grads, self.opt_state, modules)
        self.opt_state = opt_state

        scale = 1
        step = self.step.read().astype(f32)
        if self.warmup > 0:
            scale *= jnp.clip(step / self.warmup, 0, 1)
        assert self.schedule == 'constant' or self.anneal > self.warmup
        prog = jnp.clip((step - self.warmup) / (self.anneal - self.warmup), 0, 1)
        if self.schedule == 'constant':
            pass
        elif self.schedule == 'linear':
            scale *= 1 - prog
        elif self.schedule == 'cosine':
            scale *= 0.5 * (1 + jnp.cos(jnp.pi * prog))
        else:
            raise NotImplementedError(self.schedule)
        updates = treemap(lambda x: x * scale, updates)

        model = eqx.apply_updates(modules, updates)

        grad_norm = optax.global_norm(grads)
        update_norm = optax.global_norm(updates)
        param_norm = optax.global_norm([x.find() for x in modules])
        isfin = jnp.isfinite
        if self.scaling:
            self._update_scale(grads, jnp.isfinite(grad_norm))
            metrics['grad_scale'] = self.grad_scale
            metrics['grad_overflow'] = (~jnp.isfinite(grad_norm)).astype(f32)
            grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
            self.step += isfin(grad_norm).astype(i32)
        else:
            check(isfin(grad_norm), f'optimizer grad norm: {{x}}', x=grad_norm)
            self.step += 1
        check(isfin(update_norm), f'optimizer updates: {{x}}', x=update_norm)
        check(isfin(param_norm), f'optimizer params: {{x}}', x=param_norm)

        metrics['loss'] = loss.mean()
        metrics['grad_norm'] = grad_norm
        metrics['update_norm'] = update_norm
        metrics['param_norm'] = param_norm
        metrics['grad_steps'] = self.step
        metrics = {f'optimizer_{k}': v for k, v in metrics.items()}
        return (metrics, aux) if has_aux else metrics


def expand_groups(groups, keys):
    if isinstance(groups, (float, int)):
        return {key: groups for key in keys}
    groups = {
        group if group.endswith('/') else f'{group}/': value
        for group, value in groups.items()}
    assignment = {}
    groupcount = collections.defaultdict(int)
    for key in keys:
        matches = [prefix for prefix in groups if key.startswith(prefix)]
        if not matches:
            raise ValueError(
                f'Parameter {key} not fall into any of the groups:\n' +
                ''.join(f'- {group}\n' for group in groups.keys()))
        if len(matches) > 1:
            raise ValueError(
                f'Parameter {key} fall into more than one of the groups:\n' +
                ''.join(f'- {group}\n' for group in groups.keys()))
        assignment[key] = matches[0]
        groupcount[matches[0]] += 1
    for group in groups.keys():
        if not groupcount[group]:
            raise ValueError(
                f'Group {group} did not match any of the {len(keys)} keys.')
    expanded = {key: groups[assignment[key]] for key in keys}
    return expanded


def scale_by_groups(groups):
    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        scales = expand_groups(groups, updates.keys())
        updates = treemap(lambda u, s: u * s, updates, scales)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_agc(clip=0.03, pmin=1e-3):
    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        def fn(param, update):
            unorm = jnp.linalg.norm(update.flatten(), 2)
            pnorm = jnp.linalg.norm(param.flatten(), 2)
            upper = clip * jnp.maximum(pmin, pnorm)
            return update * (1 / jnp.maximum(1.0, unorm / upper))

        updates = treemap(fn, params, updates)
        return updates, ()

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):
    def init_fn(params):
        nu = treemap(lambda t: jnp.zeros_like(t, f32), params)
        step = jnp.zeros((), i32)
        return step, nu

    def update_fn(updates, state, params=None):
        step, nu = state
        step = optax.safe_int32_increment(step)
        nu = treemap(lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
        nu_hat = optax.bias_correction(nu, beta, step)
        updates = treemap(lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
        return updates, (step, nu)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):
    def init_fn(params):
        mu = treemap(lambda t: jnp.zeros_like(t, f32), params)
        step = jnp.zeros((), i32)
        return step, mu

    def update_fn(updates, state, params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu_nesterov = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu_nesterov, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)

    return optax.GradientTransformation(init_fn, update_fn)
