from distutils.core import setup

setup(
    name='ssm_mot',
    version='0.0.1',
    packages=['loss', 'model', 'dataloader', 'tracking'],
    url='',
    license='',
    author='Sai Vignesh Abishek',
    author_email='',
    description='',
    install_requires=[
        'numpy==1.24.4',
        'hydra-core',
        'wandb',
    ]
)
