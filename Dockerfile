FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN apt-get update
RUN apt-get install -y vim git

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

ADD .ssh /home/user/.ssh
RUN chown -R user:user /home/user/.ssh ; exit 0
RUN chmod 600 /home/user/.ssh/id_rsa ; exit 0

RUN chown user /opt/conda

USER user
COPY requirements.txt /tmp

ENV PATH $PATH:/home/user/.local/bin

RUN pip install --user -r /tmp/requirements.txt