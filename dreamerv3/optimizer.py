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

ENABLE_CHECKS = False
COMPUTE_DTYPE = f32
SCOPE = ''


def sg(x):
    return treemap(jax.lax.stop_gradient, x)


def check(predicate, message, **kwargs):
    if ENABLE_CHECKS:
        checkify.check(predicate, message, **kwargs)


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

    @property
    def name(self):
        """The name of this module instance as a string."""
        return 'optimizer'

    @property
    def path(self):
        """The unique name scope of this module instance as a string."""
        return 'optimizer'

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
        self.step = nj.Variable(jnp.array, 0, i32, name='step')
        self.scaling = (COMPUTE_DTYPE == jnp.float16)
        if self.scaling:
            self.chain = optax.apply_if_finite(
                self.chain, max_consecutive_errors=1000)
            self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
            self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')
        self.once = True

    def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
        def wrapped(*args, **kwargs):
            outs = lossfn(*args, **kwargs)
            loss, aux = outs if has_aux else (outs, None)
            assert loss.dtype == f32, (self.name, loss.dtype)
            assert loss.shape == (), (self.name, loss.shape)
            if self.scaling:
                loss *= sg(self.grad_scale.read())
            return loss, aux

        metrics = {}

        def wrapper(*args, **kwargs):
            accessed, modified = _prerun(fun, *args, **kwargs)

            strs = []
            for key in keys:
                if isinstance(key, Module):
                    matches = key.find()
                if isinstance(key, str):
                    pattern = re.compile(f'^{key}(/.*|$)')
                    matches = [k for k in context() if pattern.match(k)]
                if not matches:
                    raise KeyError(
                        f"Gradient key '{key}' did not match any state entries. "
                        'List existing entries using print(nj.context().keys()).')
                strs += matches
            existing = context().keys()
            assert all(key in existing for key in strs), (strs, existing)
            x1 = {k: v for k, v in context().items() if k in strs}
            x2 = {k: v for k, v in context().items() if k not in strs}
            assert x1

            for key in x1.keys():
                if key not in accessed:
                    raise RuntimeError(
                        f"Trying to compute gradient with respect to key '{key}' "
                        'but the differentiated function does not access it.\n'
                        f'Accessed keys: {list(accessed)}\n'
                        f'Gradient keys: {list(strs)}')
            x1 = {k: v for k, v in x1.items() if k in accessed}
            x2 = {k: v for k, v in x2.items() if k in accessed}

            def forward(x1, x2, *args, **kwargs):
                before = {**x1, **x2}
                state, (y, aux) = fun(before, *args, create=False, **kwargs)
                changes = {k: v for k, v in state.items() if k in modified}
                return y, (changes, aux)

            backward = jax.value_and_grad(forward, has_aux=True)

            (y, (changes, aux)), dx = backward(
                x1, x2, *args, seed=seed(None, True), **kwargs)
            if context().modify:
                context().update(changes)
            return (y, x1, dx, aux) if has_aux else (y, x1, dx)

        loss, grads = eqx.filter_value_and_grad(wrapped)(modules, *args, **kwargs)
        updates, opt_state = self.chain.update(grads, opt_state, modules)
        model = eqx.apply_updates(modules, updates)

        if self.scaling:
            loss /= self.grad_scale.read()
        if not isinstance(modules, (list, tuple)):
            modules = [modules]
        counts = {k: int(np.prod(v.shape)) for k, v in params.items()}
        if self.once:
            self.once = False
            prefs = []
            for key in counts:
                parts = key.split('/')
                prefs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), 2))]
            subcounts = {
                prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
                for prefix in set(prefs)}
            print(f'Optimizer {self.name} has {sum(counts.values()):,} variables:')
            for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
                print(f'{count:>14,} {prefix}')

        if parallel():
            grads = treemap(lambda x: jax.lax.pmean(x, 'i'), grads)
        if self.scaling:
            invscale = 1.0 / self.grad_scale.read()
            grads = treemap(lambda x: x * invscale, grads)
        optstate = self.get('state', self.chain.init, params)
        updates, optstate = self.chain.update(grads, optstate, params)
        self.put('state', optstate)

        if self.details:
            metrics.update(self._detailed_stats(optstate, params, updates, grads))

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

        nj.context().update(optax.apply_updates(params, updates))
        grad_norm = optax.global_norm(grads)
        update_norm = optax.global_norm(updates)
        param_norm = optax.global_norm([x.find() for x in modules])
        isfin = jnp.isfinite
        if self.scaling:
            self._update_scale(grads, jnp.isfinite(grad_norm))
            metrics['grad_scale'] = self.grad_scale.read()
            metrics['grad_overflow'] = (~jnp.isfinite(grad_norm)).astype(f32)
            grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
            self.step.write(self.step.read() + isfin(grad_norm).astype(i32))
        else:
            check(isfin(grad_norm), f'{self.path} grad norm: {{x}}', x=grad_norm)
            self.step.write(self.step.read() + 1)
        check(isfin(update_norm), f'{self.path} updates: {{x}}', x=update_norm)
        check(isfin(param_norm), f'{self.path} params: {{x}}', x=param_norm)

        metrics['loss'] = loss.mean()
        metrics['grad_norm'] = grad_norm
        metrics['update_norm'] = update_norm
        metrics['param_norm'] = param_norm
        metrics['grad_steps'] = self.step.read()
        metrics['param_count'] = jnp.array(sum(counts.values()))
        metrics = {f'{self.name}_{k}': v for k, v in metrics.items()}
        return (metrics, aux) if has_aux else metrics

    def _update_scale(self, grads, finite):
        keep = (finite & (self.good_steps.read() < 1000))
        incr = (finite & (self.good_steps.read() >= 1000))
        decr = ~finite
        self.good_steps.write(
            keep.astype(i32) * (self.good_steps.read() + 1))
        self.grad_scale.write(jnp.clip(
            keep.astype(f32) * self.grad_scale.read() +
            incr.astype(f32) * self.grad_scale.read() * 2 +
            decr.astype(f32) * self.grad_scale.read() / 2,
            1e-4, 1e5))
        return finite

    def _detailed_stats(self, optstate, params, updates, grads):
        groups = {
            'all': r'.*',
            'enc': r'/enc/.*',
            'dec': r'/dec/.*',
            'dyn': r'/dyn/.*',
            'con': r'/con/.*',
            'rew': r'/rew/.*',
            'actor': r'/actor/.*',
            'critic': r'/critic/.*',
            'out': r'/out/kernel$',
            'repr': r'/repr_logit/kernel$',
            'prior': r'/prior_logit/kernel$',
            'offset': r'/offset$',
            'scale': r'/scale$',
        }
        metrics = {}
        stddev = None
        for state in getattr(optstate, 'inner_state', optstate):
            if isinstance(state, optax.ScaleByAdamState):
                corr = 1 / (1 - 0.999 ** state.count)
                stddev = treemap(lambda x: jnp.sqrt(x * corr), state.nu)
        for name, pattern in groups.items():
            keys = [k for k in params if re.search(pattern, k)]
            ps = [params[k] for k in keys]
            us = [updates[k] for k in keys]
            gs = [grads[k] for k in keys]
            if not ps:
                continue
            metrics.update({f'{k}/{name}': v for k, v in dict(
                param_count=jnp.array(np.sum([np.prod(x.shape) for x in ps])),
                param_abs_max=jnp.stack([jnp.abs(x).max() for x in ps]).max(),
                param_abs_mean=jnp.stack([jnp.abs(x).mean() for x in ps]).mean(),
                param_norm=optax.global_norm(ps),
                update_abs_max=jnp.stack([jnp.abs(x).max() for x in us]).max(),
                update_abs_mean=jnp.stack([jnp.abs(x).mean() for x in us]).mean(),
                update_norm=optax.global_norm(us),
                grad_norm=optax.global_norm(gs),
            ).items()})
            if stddev is not None:
                sc = [stddev[k] for k in keys]
                pr = [
                    jnp.abs(x) / jnp.maximum(1e-3, jnp.abs(y)) for x, y in zip(us, ps)]
                metrics.update({f'{k}/{name}': v for k, v in dict(
                    scale_abs_max=jnp.stack([jnp.abs(x).max() for x in sc]).max(),
                    scale_abs_min=jnp.stack([jnp.abs(x).min() for x in sc]).min(),
                    scale_abs_mean=jnp.stack([jnp.abs(x).mean() for x in sc]).mean(),
                    prop_max=jnp.stack([x.max() for x in pr]).max(),
                    prop_min=jnp.stack([x.min() for x in pr]).min(),
                    prop_mean=jnp.stack([x.mean() for x in pr]).mean(),
                ).items()})
        return metrics


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
