import jax
import jax.numpy as jnp
from functools import partial

def init_mlp(key, layers):
    """Inicializa los pesos de un MLP usando He Initialization para ReLU."""
    weights = []
    keys = jax.random.split(key, len(layers) - 1)
    for i in range(len(layers) - 1):
        w = jax.random.normal(keys[i], (layers[i], layers[i+1])) * jnp.sqrt(2.0 / layers[i])
        b = jnp.zeros((layers[i+1],))
        if i == len(layers) - 2: # Near-zero initial velocity command
            w = w * 0.001
        weights.append((w, b))
    return tuple(weights)

def mlp_forward(obs, weights, mutations=None):
    x = obs
    for i in range(len(weights)):
        W, b = weights[i]
        
        x_base = jnp.dot(x, W)
        
        if mutations is not None:
            mut_w, mut_b = mutations[i]
            if isinstance(mut_w, tuple): # Low rank
                A, B = mut_w
                x_mut = jnp.dot(jnp.dot(x, A), B.T)
                x = x_base + x_mut + b + mut_b
            else: # Dense
                x = x_base + jnp.dot(x, mut_w) + b + mut_b
        else:
            x = x_base + b
            
        if i < len(weights) - 1:
            x = jax.nn.leaky_relu(x)
    return jnp.tanh(x)

def generate_mlp_mutation(key, layers, r=1, sigma=0.1):
    mutations = []
    keys = jax.random.split(key, (len(layers) - 1) * 3)
    idx = 0
    for i in range(len(layers) - 1):
        in_dim, out_dim = layers[i], layers[i+1]
        k1, k2, k3 = keys[idx:idx+3]
        idx += 3
        E_b = sigma * jax.random.normal(k3, (out_dim,))
        if r <= 0: # Mutación Densa (Full rank)
            E_w = sigma * jax.random.normal(k1, (in_dim, out_dim))
            mutations.append((E_w, E_b))
        else: # Hyperscale ES (Low rank)
            A = jax.random.normal(k1, (in_dim, r)) * (sigma / jnp.sqrt(r))
            B = jax.random.normal(k2, (out_dim, r))
            mutations.append(((A, B), E_b))
    return tuple(mutations)

@partial(jax.jit, static_argnums=(1, 2, 4))
def generate_and_apply_mutations(subkey, layers, rank, sigma, num_drones):
    """One low-rank mutation per WORLD, shared by its K agents."""
    if num_drones < 2 or num_drones % 2:
        raise ValueError("ES population must be even")
    keys = jax.random.split(subkey, num_drones // 2)
    halves = jax.vmap(generate_mlp_mutation, in_axes=(0, None, None, None))(keys,layers,rank,sigma)
    result=[]
    for matrix,bias in halves:
        if rank > 0:
            A,B=matrix
            matrix=(jnp.concatenate([A,-A]),jnp.concatenate([B,B]))
        else:
            matrix=jnp.concatenate([matrix,-matrix])
        result.append((matrix,jnp.concatenate([bias,-bias])))
    return tuple(result)


def centered_ranks(rewards):
    from jax.scipy.stats import rankdata
    return (rankdata(rewards,method='average')-1)/(rewards.size-1)-.5


def es_update(weights,mutations,rewards,m,v,iteration,alpha,sigma,cfg):
    """Original Eggroll estimator + rank shaping + clipped Adam ascent."""
    ranks=centered_ranks(rewards)
    gradients=[]
    for (matrix,bias),(W,b) in zip(mutations,weights):
        if cfg.mutation_rank>0:
            A,B=matrix
            gW=jnp.einsum('n,nir,nor->io',ranks,A,B)/(rewards.size*sigma**2)
        else:
            gW=jnp.tensordot(ranks,matrix,axes=1)/(rewards.size*sigma**2)
        gb=jnp.tensordot(ranks,bias,axes=1)/(rewards.size*sigma**2)
        gradients.append((gW-cfg.weight_decay*W,gb-cfg.weight_decay*b))
    gradients=tuple(gradients)
    norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree_util.tree_leaves(gradients)))
    gradients=jax.tree_util.tree_map(lambda g:g*jnp.minimum(1.,cfg.grad_clip/(norm+1e-8)),gradients)
    m=jax.tree_util.tree_map(lambda a,g:cfg.beta1*a+(1-cfg.beta1)*g,m,gradients)
    v=jax.tree_util.tree_map(lambda a,g:cfg.beta2*a+(1-cfg.beta2)*g*g,v,gradients)
    mh=jax.tree_util.tree_map(lambda x:x/(1-cfg.beta1**iteration),m)
    vh=jax.tree_util.tree_map(lambda x:x/(1-cfg.beta2**iteration),v)
    weights=jax.tree_util.tree_map(lambda w,a,b:w+alpha*a/(jnp.sqrt(b)+cfg.eps_adam),weights,mh,vh)
    return weights,m,v,norm
