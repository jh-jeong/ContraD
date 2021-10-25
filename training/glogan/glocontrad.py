import torch
import torch.nn.functional as F

from third_party.gather_layer import GatherLayer
from training.criterion import nt_xent


def _sample_generator(G, num_samples, enable_grad=True):
    latent_samples = G.sample_latent(num_samples)
    with torch.set_grad_enabled(enable_grad):
        generated_data = G(latent_samples)
    return generated_data

def supcon_fake(out1, out2, others, temperature, distributed=False):
    if distributed:
        out1 = torch.cat(GatherLayer.apply(out1), dim=0)
        out2 = torch.cat(GatherLayer.apply(out2), dim=0)
        others = torch.cat(GatherLayer.apply(others), dim=0)
    N = out1.size(0)

    _out = [out1, out2, others]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)
    mask[2*N:,2*N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[2*N:]
    mask = mask[2*N:]
    mask = mask / mask.sum(1, keepdim=True)

    lsm = F.log_softmax(sim_matrix, dim=1)
    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()
    return d_loss

def loss_D_fn(P, D, options, images, gen_images):
    assert images.size(0) == gen_images.size(0)
    gen_images = gen_images.detach()
    N = images.size(0)

    cat_images = torch.cat([images, images, gen_images], dim=0)
    d_all, aux = D(P.augment_fn(cat_images), sg_linear=True, projection=True, projection2=True)
    views = aux['projection']
    views = F.normalize(views)
    view1, view2, others = views[:N], views[N:2*N], views[2*N:]
    simclr_loss = nt_xent(view1, view2, temperature=P.temp, distributed=P.distributed, normalize=False)

    reals = aux['projection2']
    reals = F.normalize(reals)
    real1, real2, fakes = reals[:N], reals[N:2*N], reals[2*N:]
    sup_loss = supcon_fake(real1, real2, fakes, temperature=P.temp, distributed=P.distributed)

    d_real, d_gen = d_all[:N], d_all[2*N:3*N]
    if options['loss'] == 'nonsat':
        d_loss = F.softplus(d_gen).mean() + F.softplus(-d_real).mean()
    elif options['loss'] == 'wgan':
        d_loss = d_gen.mean() - d_real.mean()
    elif options['loss'] == 'hinge':
        d_loss = F.relu(1. + d_gen, inplace=True).mean() + F.relu(1. - d_real, inplace=True).mean()
    elif options['loss'] == 'lsgan':
        d_loss_real = ((d_real - 1.0) ** 2).mean()
        d_loss_fake = (d_gen ** 2).mean()
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
    else:
        raise NotImplementedError()

    return simclr_loss + P.lbd_a * sup_loss, {
        "penalty": d_loss,
        "d_real": d_real.mean(),
        "d_gen": d_gen.mean(),
    }



def loss_G_fn(P, D, options, images, gen_images):
    d_gen = D(P.augment_fn(gen_images))
    if options['loss'] == 'nonsat':
        g_loss = F.softplus(-d_gen).mean()
    elif options['loss'] == 'lsgan':
        g_loss = 0.5 * ((d_gen - 1.0) ** 2).mean()
    else:
        g_loss = -d_gen.mean()

    return g_loss


def loss_Z_fn(P, D, G, z, reconstructive_loss, images):
    N = images.size(0)

    z_r1 = z[:,0,:].reshape(N,-1)
    z_r2 = z[:,1,:].reshape(N,-1)
    gen_1 = G(z_r1)
    gen_2 = G(z_r2)


    real_views = P.augment_fn(torch.cat([images, images], dim=0))
    gen_views = torch.cat([gen_1, gen_2], dim=0)

    recon_loss = reconstructive_loss(real_views, gen_views)
    simclr_loss = nt_xent(z_r1, z_r2, temperature=P.z_temp, distributed=P.distributed, normalize=True)

    return recon_loss + P.z_contraloss_weight*simclr_loss
