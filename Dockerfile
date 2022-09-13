FROM continuumio/anaconda3
EXPOSE 8501
WORKDIR /
COPY . .

#RUN apt-get install python3-pip -y

RUN conda create --name d4 --file requirements.txt
RUN conda activate d4
RUN data/download_dataset.sh
RUN python data/sparse_molecular_dataset.py

CMD python main.py
