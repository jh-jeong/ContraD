
def get_architecture(architecture, image_size, P=None):
    if architecture == 'sndcgan':
        from models.gan.sndcgan import G_SNDCGAN, D_SNDCGAN
        generator = G_SNDCGAN(image_size=image_size)
        discriminator = D_SNDCGAN(image_size=image_size, mlp_linear=True, d_hidden=512)

    elif architecture == 'snresnet18':
        from models.gan.sndcgan import G_SNDCGAN
        from models.gan.snresnet import D_SNResNet18
        generator = G_SNDCGAN(image_size=image_size)
        discriminator = D_SNResNet18(mlp_linear=True, d_hidden=1024)

    elif architecture == 'stylegan2':
        from models.gan.stylegan2.generator import Generator
        from models.gan.stylegan2.discriminator import ResidualDiscriminatorP
        resolution = image_size[0]
        generator = Generator(size=resolution, n_mlp=8, small32=True)
        discriminator = ResidualDiscriminatorP(size=resolution, small32=True,
                                               mlp_linear=True, d_hidden=512)
    elif architecture == 'stylegan2_512':
        from models.gan.stylegan2.generator import Generator
        from models.gan.stylegan2.discriminator import ResidualDiscriminatorP
        resolution = image_size[0]
        generator = Generator(size=resolution, n_mlp=8, channel_multiplier=1.0)
        discriminator = ResidualDiscriminatorP(size=resolution, channel_multiplier=1.0,
                                               mlp_linear=True, d_hidden=512)
    else:
        raise NotImplementedError()

    return generator, discriminator