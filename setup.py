# Copyright (c) 2024 Javad Komijani


from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


packages = [
        'diffusion_model',
        'diffusion_model.device',
        'diffusion_model.nn',
        'diffusion_model.nn.gauge',
        'diffusion_model.nn.scalar'
        ]

package_dir = {
        'diffusion_model': 'src',
        'diffusion_model.device': 'src/device',
        'diffusion_model.nn': 'src/nn',
        'diffusion_model.nn.gauge': 'src/nn/gauge',
        'diffusion_model.nn.scalar': 'src/nn/scalar'
        }

setup(name='diffusion_model',
      version='1.0.0',
      description='Diffusion models for lattice field theory',
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/diffusion_model',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      install_requires=['numpy>=2.1.0', 'torch>=2.5.0'],
      zip_safe=False
      )
