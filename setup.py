import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='cmpc2',
    version='0.0.1',
    author='wuyangchen',
    author_email='chenwuyangcn@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/wuyangchen97/cmpc2',
    project_urls = {
        "Bug Tracker": "https://github.com/wuyangchen97/cmpc2/issues"
    },
    license='MIT',
    packages=['cmpc2'],
    install_requires=['numpy',
    'torch','webrtcvad'],
)