FROM spack/ubuntu-jammy:latest as bootstrap

# Input spack environment file
ARG spack_yaml

# Install ubuntu packages
RUN apt-get -yqq update \
 && apt-get -yqq install --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        file \
        fonts-texgyre \
        g++ \
        gcc \
        gfortran \
        git \
        gnupg2 \
        iproute2 \
        latexmk \
        locales \
        lua-posix \
        make \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-venv \
        tex-gyre \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-plain-generic \
        unzip \
        csh \
 && locale-gen en_US.UTF-8 \
 && pip3 install boto3 \
 && rm -rf /var/lib/apt/lists/*

# Build stage with Spack pre-installed and ready to be used
FROM bootstrap as builder

# Install the software, remove unnecessary deps
RUN mkdir /opt/spack-environment
COPY $spack_yaml /opt/spack-environment/spack.yaml
RUN cd /opt/spack-environment \
 && spack env activate . \
 && spack install --fail-fast \
 && spack clean --all \
 && spack gc -y

# Modifications to the environment that are necessary to run
RUN cd /opt/spack-environment && \
    spack env activate --sh -d . >> /etc/profile.d/z10_spack_environment.sh

# Update path
ENV PATH="/opt/view/bin:$PATH"

# Allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Suppress HWLOC errors
ENV HWLOC_HIDE_ERRORS=2

ENTRYPOINT ["/bin/bash", "--rcfile", "/etc/profile", "-l"]
