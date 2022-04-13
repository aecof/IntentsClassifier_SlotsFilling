FROM pytorch/pytorch


ADD Intent_Slot_Filling Intent_Slot_Filling

RUN apt-get update
RUN apt-get -y install vim
RUN apt-get -y install iputils-ping
RUN apt-get -y install curl
RUN pip install "ray[serve]"
RUN pip install unidecode
RUN pip install joblib
RUN pip install requests
RUN pip install -e Intent_Slot_Filling
WORKDIR Intent_Slot_Filling




