import architectures.resnet18
import architectures.resnet50
import architectures.bninception


def select(arch, opt):
    if 'resnet18' in arch:
        return resnet18.Network(opt)
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)
